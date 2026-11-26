use crate::digest;
use dkr_kernel_registry::{Registry, RegistryKey};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ops::Deref,
    path::{Component, Path, PathBuf},
    sync::Arc,
};
#[cfg(any(feature = "cuda", test))]
use std::{
    sync::{Mutex, Weak},
};

pub const TYPED_ABI: &str = "dkr.typed.v1";
pub const MODULE_SCHEMA: &str = "DKR-KERNEL-MODULE-V1";
pub const MAX_MANIFEST_BYTES: u64 = 256 * 1024;
pub const MAX_KERNEL_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_ARGUMENTS: usize = 32;
pub const MAX_TENSOR_BYTES: usize = 1024 * 1024 * 1024;
pub const MAX_DISCOVERY_DEPTH: usize = 8;
pub const MAX_DISCOVERED_PACKAGES: usize = 512;
pub const MANIFEST_FILENAME: &str = "manifest.json";
/// Independent physical discovery and logical dispatch bounds.
pub const MAX_PROFILES_PER_SHARD: usize = 64;
pub const MAX_CATALOG_PROFILES: usize = 4096;
pub const SHARD_SCHEMA: &str = "DKR-KERNEL-SHARD-V1";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct KernelShard {
    schema: String,
    profiles: Vec<KernelManifest>,
}

/// Embedded images remain zero-copy; external profiles share verified storage.
#[derive(Debug)]
pub enum SharedImage<T: ?Sized + 'static> {
    Embedded(&'static T),
    External(Arc<T>),
}
impl<T: ?Sized + 'static> Clone for SharedImage<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Embedded(bytes) => Self::Embedded(*bytes),
            Self::External(bytes) => Self::External(Arc::clone(bytes)),
        }
    }
}
impl<T: ?Sized + 'static> Deref for SharedImage<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Embedded(bytes) => *bytes,
            Self::External(bytes) => bytes.as_ref(),
        }
    }
}
impl<T: ?Sized + 'static> AsRef<T> for SharedImage<T> {
    fn as_ref(&self) -> &T { self.deref() }
}

#[derive(Default)]
struct ImageCache {
    ptx: BTreeMap<PathBuf, (String, SharedImage<str>)>,
    cubin: BTreeMap<PathBuf, (String, SharedImage<[u8]>)>,
}

fn bounded_read(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let metadata = std::fs::metadata(path)?;
    ensure!(metadata.is_file() && metadata.len() <= limit, "{label} budget exceeded");
    let file = std::fs::File::open(path)?;
    ensure!(file.metadata()?.is_file() && file.metadata()?.len() <= limit, "{label} budget exceeded");
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "{label} budget exceeded");
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    F64,
    F32,
    F16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
}

impl ElementType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::Fp8E4m3 | Self::Fp8E5m2 | Self::I8 | Self::U8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Parameter {
    Tensor {
        name: String,
        dtype: ElementType,
        rank: usize,
        access: Access,
        #[serde(default)]
        representation: Option<String>,
    },
    Scalar {
        name: String,
        dtype: ElementType,
    },
}

impl Parameter {
    fn name(&self) -> &str {
        match self {
            Self::Tensor { name, .. } | Self::Scalar { name, .. } => name,
        }
    }

    fn ptx_width(&self) -> usize {
        match self {
            Self::Tensor { .. } => 64,
            Self::Scalar { dtype, .. } => dtype.size_bytes() * 8,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelManifest {
    pub schema: String,
    pub abi: String,
    pub operation: String,
    pub implementation_id: String,
    pub artifact: PathBuf,
    pub sha256: String,
    /// Optional CUDA machine image assembled from the paired, ABI-checked PTX.
    #[serde(default)]
    pub cubin_artifact: Option<PathBuf>,
    #[serde(default)]
    pub cubin_sha256: Option<String>,
    pub entry: String,
    pub target_cc: u32,
    pub min_cc: u32,
    pub max_cc: u32,
    pub block: [u32; 3],
    pub shared_mem_bytes: u32,
    pub arguments: Vec<Parameter>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Argument {
    Tensor {
        dtype: ElementType,
        shape: Vec<usize>,
        contiguous: bool,
        start_offset: usize,
        representation: Option<String>,
    },
    Scalar {
        dtype: ElementType,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchRequest {
    pub operation: String,
    pub cc: u32,
    pub args: Vec<Argument>,
    pub grid: [u32; 3],
    pub max_dynamic_shared_mem_bytes: u32,
}

#[derive(Debug)]
pub struct VerifiedKernel {
    pub manifest: KernelManifest,
    pub ptx: SharedImage<str>,
    pub cubin: Option<SharedImage<[u8]>>,
    pub manifest_sha256: String,
}

fn is_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-')
}

fn param_bit_width(parameter: &str) -> Option<usize> {
    parameter.split_whitespace().find_map(|part| {
        let ty = part.trim_end_matches(',');
        let width = [".u", ".s", ".b", ".f"]
            .iter()
            .find_map(|prefix| ty.strip_prefix(prefix))?;
        match width {
            "8" => Some(8),
            "16" => Some(16),
            "32" => Some(32),
            "64" => Some(64),
            _ => None,
        }
    })
}

// CUDA cubins are little-endian ELF64 files with EM_CUDA (190).
// Admission of the actual image and entry is additionally checked by the CUDA driver.
fn cubin_elf_header(bytes: &[u8], target_cc: u32) -> bool {
    bytes.len() >= 64
        && bytes.starts_with(b"\x7fELF")
        && bytes[4] == 2
        && bytes[5] == 1
        && u16::from_le_bytes([bytes[18], bytes[19]]) == 190
        // For pre-Blackwell CUDA, EF_CUDA_SM is e_flags & 0xff (LLVM ELF.h).
        // The next byte contains independent CUDA flags, NOT the Pascal SM.
        && (u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]) & 0xff) == target_cc
}

fn entry_signature(ptx: &str, entry: &str) -> Result<Vec<usize>> {
    for (index, _) in ptx.match_indices(".entry") {
        let body = ptx[index + ".entry".len()..].trim_start();
        let Some(after) = body.strip_prefix(entry) else {
            continue;
        };
        let Some(params) = after.trim_start().strip_prefix('(') else {
            continue;
        };
        let text = params.split(')').next().context("unterminated PTX entry")?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        return text
            .split(',')
            .map(|p| param_bit_width(p).context("unsupported PTX entry parameter"))
            .collect();
    }
    bail!("PTX entry symbol missing")
}

impl KernelManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == MODULE_SCHEMA && self.abi == TYPED_ABI,
            "unsupported typed module ABI/schema"
        );
        ensure!(
            is_identifier(&self.operation) && is_identifier(&self.implementation_id),
            "invalid operation identity"
        );
        ensure!(
            is_identifier(&self.entry) && !self.entry.contains('.') && !self.entry.contains('-'),
            "invalid CUDA entry symbol"
        );
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid SHA256"
        );
        ensure!(
            self.target_cc >= 50 && self.target_cc <= self.min_cc && self.min_cc <= self.max_cc,
            "invalid compute capability bounds"
        );
        ensure!(
            self.block.iter().all(|&v| v > 0)
                && self
                    .block
                    .iter()
                    .try_fold(1u32, |n, &v| n.checked_mul(v))
                    .is_some_and(|v| v <= 1024),
            "invalid launch block"
        );
        ensure!(
            self.shared_mem_bytes <= 96 * 1024,
            "shared memory budget exceeded"
        );
        ensure!(
            !self.arguments.is_empty() && self.arguments.len() <= MAX_ARGUMENTS,
            "invalid typed argument count"
        );
        let mut names = std::collections::HashSet::new();
        for arg in &self.arguments {
            ensure!(
                is_identifier(arg.name()) && names.insert(arg.name()),
                "duplicate or invalid argument name"
            );
            if let Parameter::Tensor {
                rank,
                representation,
                ..
            } = arg
            {
                ensure!((1..=8).contains(rank), "unsupported tensor rank");
                ensure!(
                    representation.as_deref().is_none_or(is_identifier),
                    "invalid packed tensor representation"
                );
            }
        }
        ensure!(
            self.artifact
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
                && self.artifact.extension().is_some_and(|e| e == "ptx"),
            "DKR-R1 requires a relative PTX source artifact"
        );
        match (&self.cubin_artifact, &self.cubin_sha256) {
            (None, None) => {}
            (Some(path), Some(hash)) => {
                ensure!(
                    path.components().all(|c| matches!(c, Component::Normal(_)))
                        && path.extension().is_some_and(|e| e == "cubin"),
                    "CUBIN artifact must be a relative .cubin path"
                );
                ensure!(
                    hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
                    "invalid CUBIN SHA256"
                );
            }
            _ => bail!("CUBIN path and digest must be declared together"),
        }
        Ok(())
    }

    pub fn admit(&self, request: &LaunchRequest) -> Result<()> {
        self.validate()?;
        ensure!(
            request.operation == self.operation,
            "operation identity mismatch"
        );
        ensure!(
            (self.min_cc..=self.max_cc).contains(&request.cc),
            "GPU architecture outside kernel contract"
        );
        ensure!(
            request.grid.iter().all(|&v| v > 0),
            "zero launch grid dimension"
        );
        ensure!(
            self.shared_mem_bytes <= request.max_dynamic_shared_mem_bytes,
            "device shared memory limit exceeded"
        );
        ensure!(
            request.args.len() == self.arguments.len(),
            "typed argument count mismatch"
        );
        let mut total_bytes = 0usize;
        for (actual, expected) in request.args.iter().zip(&self.arguments) {
            match (actual, expected) {
                (Argument::Scalar { dtype: got }, Parameter::Scalar { dtype, .. })
                    if got == dtype => {}
                (
                    Argument::Tensor {
                        dtype: got,
                        shape,
                        contiguous,
                        start_offset,
                        representation: got_repr,
                    },
                    Parameter::Tensor {
                        dtype,
                        rank,
                        representation,
                        ..
                    },
                ) if got == dtype
                    && got_repr == representation
                    && shape.len() == *rank
                    && *contiguous
                    && *start_offset == 0 =>
                {
                    let elements = shape
                        .iter()
                        .try_fold(1usize, |n, &d| if d > 0 { n.checked_mul(d) } else { None })
                        .context("invalid tensor shape or overflow")?;
                    let bytes = elements
                        .checked_mul(dtype.size_bytes())
                        .context("tensor byte count overflow")?;
                    total_bytes = total_bytes
                        .checked_add(bytes)
                        .context("aggregate tensor budget overflow")?;
                    ensure!(
                        total_bytes <= MAX_TENSOR_BYTES,
                        "aggregate tensor budget exceeded"
                    );
                }
                _ => bail!(
                    "argument type, layout or rank mismatch for {}",
                    expected.name()
                ),
            }
        }
        Ok(())
    }
}

fn parse_profiles(bytes: &[u8]) -> Result<Vec<(KernelManifest, String)>> {
    // Preserve byte-exact V1 identity. Hash each shard profile
    // independently of its neighbours and of shard ordering.
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    let profiles = if value.get("schema").and_then(|v| v.as_str()) == Some(SHARD_SCHEMA) {
        let shard: KernelShard = serde_json::from_slice(bytes)?;
        ensure!(shard.schema == SHARD_SCHEMA && !shard.profiles.is_empty()
            && shard.profiles.len() <= MAX_PROFILES_PER_SHARD,
            "kernel shard profile count exceeded or empty");
        shard.profiles.into_iter().map(|manifest| {
            let hash = digest(&serde_json::to_vec(&manifest)?);
            Ok((manifest, hash))
        }).collect::<Result<Vec<_>>>()?
    } else {
        vec![(serde_json::from_slice::<KernelManifest>(bytes)?, digest(bytes))]
    };
    Ok(profiles)
}

impl VerifiedKernel {
    /// Validate every profile in one physical package (V1 or a shard).
    pub fn read_package(path: &Path) -> Result<Vec<Self>> {
        let path = path.canonicalize()?;
        let bytes = bounded_read(&path, MAX_MANIFEST_BYTES, "manifest")?;
        let mut cache = ImageCache::default();
        let mut ids = std::collections::BTreeSet::new();
        parse_profiles(&bytes)?.into_iter().map(|(manifest, hash)| {
            ensure!(ids.insert((manifest.operation.clone(), manifest.implementation_id.clone(), manifest.target_cc)),
                "duplicate operation/implementation/architecture identity");
            Self::read_profile(&path, manifest, hash, &mut cache)
        }).collect()
    }

    pub fn read(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let bytes = bounded_read(&path, MAX_MANIFEST_BYTES, "manifest")?;
        let manifest = serde_json::from_slice(&bytes)?;
        Self::read_profile(&path, manifest, digest(&bytes), &mut ImageCache::default())
    }

    fn read_profile(path: &Path, manifest: KernelManifest, manifest_sha256: String,
                    cache: &mut ImageCache) -> Result<Self> {
        manifest.validate()?;
        let root = path.parent().context("manifest directory missing")?;
        let artifact_path = root.join(&manifest.artifact).canonicalize()?;
        ensure!(artifact_path.starts_with(root), "kernel artifact escapes package directory");
        if !cache.ptx.contains_key(&artifact_path) {
            let bytes = bounded_read(&artifact_path, MAX_KERNEL_BYTES, "kernel artifact")?;
            let hash = digest(&bytes);
            ensure!(hash.eq_ignore_ascii_case(&manifest.sha256), "kernel artifact digest mismatch");
            let ptx = SharedImage::External(Arc::<str>::from(String::from_utf8(bytes)?));
            cache.ptx.insert(artifact_path.clone(), (hash, ptx));
        }
        let (hash, ptx) = &cache.ptx[&artifact_path];
        ensure!(hash.eq_ignore_ascii_case(&manifest.sha256), "kernel artifact digest mismatch");
        let ptx = ptx.clone();
        let tokens: Vec<_> = ptx.split_whitespace().collect();
        ensure!(
            tokens.windows(2).any(|t| t == [".address_size", "64"]),
            "PTX address width mismatch"
        );
        let target = format!("sm_{}", manifest.target_cc);
        ensure!(
            tokens
                .windows(2)
                .any(|t| t[0] == ".target" && t[1].trim_end_matches(',') == target),
            "PTX target architecture mismatch"
        );
        let widths = entry_signature(&ptx, &manifest.entry)?;
        let expected: Vec<usize> = manifest
            .arguments
            .iter()
            .map(Parameter::ptx_width)
            .collect();
        ensure!(
            widths == expected,
            "PTX entry arguments differ from typed ABI"
        );
        // Verify the complete paired image before publishing a module to the catalog.
        // No fallback may conceal a corrupt or mismatched CUBIN package.
        let cubin = if let Some(name) = &manifest.cubin_artifact {
            let path = root.join(name).canonicalize()?;
            ensure!(path.starts_with(root), "CUBIN escapes package directory");
            if !cache.cubin.contains_key(&path) {
                let bytes = bounded_read(&path, MAX_KERNEL_BYTES, "CUBIN artifact")?;
                let hash = digest(&bytes);
                ensure!(hash.eq_ignore_ascii_case(manifest.cubin_sha256.as_deref().unwrap_or("")),
                    "CUBIN digest mismatch");
                let image = SharedImage::External(Arc::<[u8]>::from(bytes));
                cache.cubin.insert(path.clone(), (hash, image));
            }
            let (hash, image) = &cache.cubin[&path];
            ensure!(hash.eq_ignore_ascii_case(manifest.cubin_sha256.as_deref().unwrap_or("")),
                "CUBIN digest mismatch");
            ensure!(cubin_elf_header(image, manifest.target_cc), "invalid CUBIN ELF header or target SM");
            Some(image.clone())
        } else {
            None
        };
        Ok(Self {
            manifest,
            ptx,
            cubin,
            manifest_sha256,
        })
    }

    pub fn module_key(&self) -> String {
        format!(
            "{}:{}:{}",
            TYPED_ABI,
            self.manifest_sha256,
            self.manifest.sha256.to_ascii_lowercase()
        )
    }
}

#[derive(Default)]
pub struct KernelCatalog {
    modules: Registry<Arc<VerifiedKernel>>,
}

impl KernelCatalog {
    pub fn from_build_verified(modules: Vec<VerifiedKernel>) -> Result<Self> {
        ensure!(modules.len() <= MAX_CATALOG_PROFILES, "logical kernel profile count exceeded");
        let mut result = Self::default();
        for module in modules {
            let key = RegistryKey::new(TYPED_ABI, &module.manifest.operation,
                &module.manifest.implementation_id, module.manifest.target_cc);
            result.modules.insert(key, Arc::new(module)).map_err(anyhow::Error::msg)?;
        }
        Ok(result)
    }

    pub fn verified_modules(&self) -> impl Iterator<Item = &Arc<VerifiedKernel>> {
        self.modules.values()
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn discover_shared(root: &Path) -> Result<Arc<Self>> {
        static CATALOGS: Mutex<BTreeMap<PathBuf, Weak<KernelCatalog>>> = Mutex::new(BTreeMap::new());
        let root = root.canonicalize()?;
        let mut catalogs = CATALOGS
            .lock()
            .map_err(|_| anyhow::anyhow!("DKR catalog cache poisoned"))?;
        if let Some(catalog) = catalogs.get(&root).and_then(Weak::upgrade) {
            return Ok(catalog);
        }
        // Keep validation inside the lock so concurrent routes publish a single snapshot.
        let catalog = Arc::new(Self::discover(&root)?);
        catalogs.retain(|_, entry| entry.strong_count() != 0);
        catalogs.insert(root, Arc::downgrade(&catalog));
        Ok(catalog)
    }

    pub fn discover(root: &Path) -> Result<Self> {
        let root = root.canonicalize()?;
        ensure!(root.is_dir(), "kernel package root is not a directory");
        let mut pending = vec![(root, 0usize)];
        let mut paths = Vec::new();
        while let Some((directory, depth)) = pending.pop() {
            for entry in std::fs::read_dir(directory)? {
                let entry = entry?;
                let kind = entry.file_type()?;
                ensure!(!kind.is_symlink(), "symlink in kernel package catalogue");
                if kind.is_dir() {
                    ensure!(
                        depth < MAX_DISCOVERY_DEPTH,
                        "kernel package nesting limit exceeded"
                    );
                    pending.push((entry.path(), depth + 1));
                } else if kind.is_file() && entry.file_name() == MANIFEST_FILENAME {
                    paths.push(entry.path());
                    ensure!(
                        paths.len() <= MAX_DISCOVERED_PACKAGES,
                        "kernel package count exceeded"
                    );
                }
            }
        }
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "no kernel manifests found in package root"
        );
        let catalog = Self::load(&paths)?;
        if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
            eprintln!("MISTRAL_DKR_CATALOG_VALIDATED manifests={} profiles={}", paths.len(), catalog.len());
        }
        Ok(catalog)
    }

    pub fn load(paths: &[PathBuf]) -> Result<Self> {
        ensure!(paths.len() <= MAX_DISCOVERED_PACKAGES, "kernel package count exceeded");
        let mut result = Self::default();
        let mut cache = ImageCache::default();
        for path in paths {
            let path = path.canonicalize()?;
            let bytes = bounded_read(&path, MAX_MANIFEST_BYTES, "manifest")?;
            let profiles = parse_profiles(&bytes)?;
            ensure!(profiles.len() <= MAX_CATALOG_PROFILES - result.len(),
                "logical kernel profile count exceeded");
            for (manifest, hash) in profiles {
                let module = Arc::new(VerifiedKernel::read_profile(&path, manifest, hash, &mut cache)?);
                let key = RegistryKey::new(TYPED_ABI, &module.manifest.operation,
                    &module.manifest.implementation_id, module.manifest.target_cc);
                result.modules.insert(key, module).map_err(|e| anyhow::anyhow!("duplicate operation/implementation/architecture identity: {e}"))?;
            }
        }
        Ok(result)
    }

    pub fn select(
        &self,
        implementation_id: &str,
        request: &LaunchRequest,
    ) -> Result<&Arc<VerifiedKernel>> {
        self.resolve_candidates(request, Some(implementation_id))
    }

    pub fn resolve(&self, request: &LaunchRequest) -> Result<&Arc<VerifiedKernel>> {
        self.resolve_candidates(request, None)
    }

    fn resolve_candidates(
        &self,
        request: &LaunchRequest,
        implementation_id: Option<&str>,
    ) -> Result<&Arc<VerifiedKernel>> {
        let result=self.modules.select(TYPED_ABI, &request.operation, implementation_id, request.cc,
            |module| module.manifest.admit(request).is_ok());
        if result.is_err(){dkr_kernel_registry::telemetry::handle(TYPED_ABI,&request.operation,"selection","",request.cc).record(dkr_kernel_registry::telemetry::Event::Rejected);}
        result.map_err(anyhow::Error::msg)
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dkr_argsort_f64_typed_abi() {
        assert_eq!(ElementType::F64.size_bytes(), 8);
        assert_eq!(serde_json::to_string(&ElementType::F64).unwrap(), "\"f64\"");
        assert_eq!(serde_json::from_str::<ElementType>("\"f64\"").unwrap(), ElementType::F64);
        let scalar = Parameter::Scalar { name: "value".into(), dtype: ElementType::F64 };
        assert_eq!(scalar.ptx_width(), 64);
        assert_eq!(param_bit_width(".param .f64 value"), Some(64));
        let tensor = Parameter::Tensor { name: "values".into(), dtype: ElementType::F64,
            rank: 1, access: Access::Read, representation: None };
        assert_eq!(tensor.ptx_width(), 64);
        let (dir,path,mut request)=fixture();
        let mut manifest:KernelManifest=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let ptx=std::fs::read_to_string(dir.path().join("fused.ptx")).unwrap().replace(".f32 alpha",".f64 alpha");
        for arg in &mut manifest.arguments {
            match arg { Parameter::Tensor {dtype,..}|Parameter::Scalar {dtype,..}=>*dtype=ElementType::F64 }
        }
        for arg in &mut request.args {
            match arg { Argument::Tensor {dtype,..}|Argument::Scalar {dtype}=>*dtype=ElementType::F64 }
        }
        manifest.sha256=digest(ptx.as_bytes());
        std::fs::write(dir.path().join("fused.ptx"),ptx).unwrap();
        std::fs::write(&path,serde_json::to_vec(&manifest).unwrap()).unwrap();
        let catalog=KernelCatalog::discover(dir.path()).unwrap();
        assert!(catalog.resolve(&request).is_ok());
        if let Argument::Tensor {dtype,..}=&mut request.args[0] {*dtype=ElementType::F32;}
        assert!(catalog.resolve(&request).is_err());
    }

    fn fixture() -> (tempfile::TempDir, PathBuf, LaunchRequest) {
        let dir = tempfile::tempdir().unwrap();
        let ptx = b".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry fused(\n .param .u64 input,\n .param .f32 alpha,\n .param .u64 output\n) { ret; }\n";
        let manifest = KernelManifest {
            schema: MODULE_SCHEMA.into(),
            abi: TYPED_ABI.into(),
            operation: "tensor.elementwise.scale.v1".into(),
            implementation_id: "pascal.f32.scalar".into(),
            artifact: "fused.ptx".into(),
            sha256: digest(ptx),
            cubin_artifact: None,
            cubin_sha256: None,
            entry: "fused".into(),
            target_cc: 61,
            min_cc: 61,
            max_cc: 61,
            block: [128, 1, 1],
            shared_mem_bytes: 0,
            arguments: vec![
                Parameter::Tensor {
                    name: "input".into(),
                    dtype: ElementType::F32,
                    rank: 2,
                    access: Access::Read,
                    representation: None,
                },
                Parameter::Scalar {
                    name: "alpha".into(),
                    dtype: ElementType::F32,
                },
                Parameter::Tensor {
                    name: "output".into(),
                    dtype: ElementType::F32,
                    rank: 2,
                    access: Access::Write,
                    representation: None,
                },
            ],
        };
        std::fs::write(dir.path().join("fused.ptx"), ptx).unwrap();
        let path = dir.path().join("manifest.json");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let tensor = Argument::Tensor {
            dtype: ElementType::F32,
            shape: vec![2, 3],
            contiguous: true,
            start_offset: 0,
            representation: None,
        };
        let request = LaunchRequest {
            operation: manifest.operation,
            cc: 61,
            args: vec![
                tensor.clone(),
                Argument::Scalar {
                    dtype: ElementType::F32,
                },
                tensor,
            ],
            grid: [1, 1, 1],
            max_dynamic_shared_mem_bytes: 48 * 1024,
        };
        (dir, path, request)
    }


    fn write_shard(path: &Path, profiles: Vec<KernelManifest>) {
        std::fs::write(path, serde_json::to_vec(&KernelShard {
            schema: SHARD_SCHEMA.into(), profiles,
        }).unwrap()).unwrap();
    }

    #[test]
    fn scalable_shards_share_images_and_preserve_explicit_variant_selection() {
        let (dir, path, request) = fixture();
        let manifest: KernelManifest = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let second = "fused_variant";
        let original = std::fs::read_to_string(dir.path().join("fused.ptx")).unwrap();
        let combined = format!("{original}\n{}", original[original.find(".visible").unwrap()..].replace("fused", second));
        std::fs::write(dir.path().join("fused.ptx"), &combined).unwrap();
        let mut a = manifest;
        a.sha256 = digest(combined.as_bytes());
        let mut b = a.clone();
        b.implementation_id = "variant.b".into();
        b.entry = second.into();
        write_shard(&path, vec![a.clone(), b.clone()]);
        let catalog = KernelCatalog::discover(dir.path()).unwrap();
        let first = catalog.select(&a.implementation_id, &request).unwrap();
        let other = catalog.select(&b.implementation_id, &request).unwrap();
        assert_eq!(other.manifest.entry, second);
        assert!(std::ptr::eq(first.ptx.as_ref(), other.ptx.as_ref()));
        assert_ne!(first.manifest_sha256, other.manifest_sha256);
        assert_eq!(VerifiedKernel::read_package(&path).unwrap().len(), 2);
        let first_hash = first.manifest_sha256.clone();
        write_shard(&path, vec![b.clone(), a.clone()]);
        assert_eq!(KernelCatalog::discover(dir.path()).unwrap()
            .select(&a.implementation_id, &request).unwrap().manifest_sha256, first_hash);
        b.entry = "absent".into();
        write_shard(&path, vec![a.clone(), b.clone()]);
        assert!(KernelCatalog::discover(dir.path()).is_err());
        b.entry = second.into();
        assert!(catalog.resolve(&request).is_err());
        b.sha256 = "0".repeat(64);
        write_shard(&path, vec![a.clone(), b]);
        assert!(KernelCatalog::discover(dir.path()).is_err());
        write_shard(&path, vec![a.clone(), a]);
        assert!(KernelCatalog::discover(dir.path()).is_err());
        assert!(VerifiedKernel::read_package(&path).is_err());
    }

    #[test]
    fn scalable_shards_fit_projected_profile_count_without_raising_discovery_limit() {
        let (dir, path, _) = fixture();
        let m: KernelManifest = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for start in (0..2151).step_by(MAX_PROFILES_PER_SHARD) {
            let sub = dir.path().join(format!("shard-{start}"));
            std::fs::create_dir(&sub).unwrap();
            std::fs::copy(dir.path().join("fused.ptx"), sub.join("fused.ptx")).unwrap();
            let rows = (start..(start + MAX_PROFILES_PER_SHARD).min(2151)).map(|i| {
                let mut row = m.clone();
                row.implementation_id = format!("variant.{i}");
                row
            }).collect();
            write_shard(&sub.join(MANIFEST_FILENAME), rows);
        }
        std::fs::remove_file(&path).unwrap();
        let catalog = KernelCatalog::discover(dir.path()).unwrap();
        assert_eq!(catalog.len(), 2151);
        assert_eq!(MAX_DISCOVERED_PACKAGES, 512);
        assert!(KernelCatalog::load(&vec![path.clone(); MAX_DISCOVERED_PACKAGES + 1]).is_err());
        for start in (2151..MAX_CATALOG_PROFILES+1).step_by(MAX_PROFILES_PER_SHARD) {
            let sub = dir.path().join(format!("extra-{start}"));
            std::fs::create_dir(&sub).unwrap();
            std::fs::copy(dir.path().join("fused.ptx"), sub.join("fused.ptx")).unwrap();
            write_shard(&sub.join(MANIFEST_FILENAME),
                (start..(start+MAX_PROFILES_PER_SHARD).min(MAX_CATALOG_PROFILES+1)).map(|i| {
                    let mut row = m.clone();
                    row.implementation_id = format!("variant.{i}");
                    row
                }).collect());
        }
        let err = KernelCatalog::discover(dir.path()).err().unwrap();
        assert!(err.to_string().contains("logical kernel profile count exceeded"));
        // Remove the extra shards before testing malformed single shards.
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name().to_string_lossy().starts_with("extra-") {
                std::fs::remove_dir_all(entry.path()).unwrap();
            }
        }
        write_shard(&path, vec![m.clone(); MAX_PROFILES_PER_SHARD + 1]);
        assert!(KernelCatalog::discover(dir.path()).is_err());
        write_shard(&path, vec![]);
        assert!(KernelCatalog::discover(dir.path()).is_err());
    }

    #[test]
    fn shares_one_verified_catalog_for_live_routes() {
        let (dir, _, request) = fixture();
        let (first, second) = std::thread::scope(|scope| {
            let first = scope.spawn(|| KernelCatalog::discover_shared(dir.path()).unwrap());
            let second = scope.spawn(|| KernelCatalog::discover_shared(&dir.path().join(".")).unwrap());
            (first.join().unwrap(), second.join().unwrap())
        });
        assert!(Arc::ptr_eq(&first, &second));
        let (other_dir, _, _) = fixture();
        let other = KernelCatalog::discover_shared(other_dir.path()).unwrap();
        assert!(!Arc::ptr_eq(&first, &other));

        let artifact = dir.path().join("fused.ptx");
        let original = std::fs::read(&artifact).unwrap();
        std::fs::write(&artifact, b"corrupted").unwrap();
        assert!(KernelCatalog::discover(dir.path()).is_err());
        assert_eq!(first.resolve(&request).unwrap().manifest.entry, "fused");
        drop(first);
        drop(second);
        assert!(KernelCatalog::discover_shared(dir.path()).is_err());
        std::fs::write(artifact, original).unwrap();
        assert_eq!(KernelCatalog::discover_shared(dir.path()).unwrap().len(), 1);
    }

    #[test]
    fn loads_without_asd_and_checks_typed_arguments() {
        let (dir, path, request) = fixture();
        let catalog = KernelCatalog::load(&[path.clone()]).unwrap();
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(dir.path().join("fused.ptx")).unwrap();
        let kernel = catalog.select("pascal.f32.scalar", &request).unwrap();
        assert_eq!(kernel.manifest.entry, "fused");
        assert_eq!(catalog.len(), 1);
        let mut bad = request.clone();
        bad.args[1] = Argument::Scalar {
            dtype: ElementType::I32,
        };
        assert!(catalog.select("pascal.f32.scalar", &bad).is_err());
        bad = request.clone();
        bad.args[0] = Argument::Tensor {
            dtype: ElementType::F32,
            shape: vec![2, 3],
            contiguous: false,
            start_offset: 0,
            representation: None,
        };
        assert!(catalog.select("pascal.f32.scalar", &bad).is_err());
        assert!(catalog.select("not-present", &request).is_err());
    }

    #[test]
    fn checks_paired_cubin_bytes_and_refuses_corruption() {
        let (dir, path, _) = fixture();
        let mut manifest: KernelManifest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut cubin = vec![0u8; 64];
        cubin[..4].copy_from_slice(b"\x7fELF");
        cubin[4] = 2; // ELF64
        cubin[5] = 1; // little endian
        cubin[18..20].copy_from_slice(&190u16.to_le_bytes()); // EM_CUDA
        cubin[48..52].copy_from_slice(&(61u32 | 0x400).to_le_bytes()); // EF_CUDA_SM61
        manifest.cubin_artifact = Some("fused.cubin".into());
        manifest.cubin_sha256 = Some(digest(&cubin));
        std::fs::write(dir.path().join("fused.cubin"), &cubin).unwrap();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let verified = VerifiedKernel::read(&path).unwrap();
        assert_eq!(verified.cubin.as_deref(), Some(cubin.as_slice()));
        std::fs::write(dir.path().join("fused.cubin"), b"corrupted").unwrap();
        assert!(VerifiedKernel::read(&path).unwrap_err().to_string().contains("digest"));

        manifest.cubin_sha256 = Some(digest(b"corrupted"));
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(VerifiedKernel::read(&path).unwrap_err().to_string().contains("ELF"));

        // A valid CUDA ELF for a different physical SM must never be selected.
        // CUDA control flags (e.g. EF_CUDA_64BIT_ADDRESS) do not alter SM.
        // The old shifted mask rejected this valid Pascal e_flags shape.
        assert!(cubin_elf_header(&cubin, 61));
        cubin[48..52].copy_from_slice(&(61u32 | 0x700).to_le_bytes());
        assert!(cubin_elf_header(&cubin, 61));
        cubin[48..52].copy_from_slice(&(75u32 | 0x400).to_le_bytes());
        manifest.cubin_sha256 = Some(digest(&cubin));
        std::fs::write(dir.path().join("fused.cubin"), &cubin).unwrap();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(VerifiedKernel::read(&path).unwrap_err().to_string().contains("target SM"));

        manifest.cubin_sha256 = None;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn rejects_tampered_artifacts_and_entry_abi() {
        let (dir, path, _) = fixture();
        std::fs::write(dir.path().join("fused.ptx"), b"bad").unwrap();
        assert!(VerifiedKernel::read(&path).is_err());
        let (_dir, path, _) = fixture();
        let mut manifest: KernelManifest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest.arguments.pop();
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(VerifiedKernel::read(&path)
            .unwrap_err()
            .to_string()
            .contains("PTX entry arguments"));
    }

    #[test]
    fn packed_formats_are_versioned_contract_data_not_host_dispatch() {
        let (_dir, path, mut request) = fixture();
        let mut manifest: KernelManifest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if let Parameter::Tensor {
            dtype,
            representation,
            ..
        } = &mut manifest.arguments[0]
        {
            *dtype = ElementType::U8;
            *representation = Some("nvfp4.u4x2".into());
        }
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        request.args[0] = Argument::Tensor {
            dtype: ElementType::U8,
            shape: vec![2, 3],
            contiguous: true,
            start_offset: 0,
            representation: Some("nvfp4.u4x2".into()),
        };
        let catalog = KernelCatalog::load(&[path]).unwrap();
        assert!(catalog.select("pascal.f32.scalar", &request).is_ok());
        if let Argument::Tensor { representation, .. } = &mut request.args[0] {
            *representation = Some("mxfp4.u4x2".into());
        }
        assert!(catalog.select("pascal.f32.scalar", &request).is_err());
    }

    #[test]
    fn rejects_bad_resource_budget_and_launch_configuration() {
        let (_dir, path, mut request) = fixture();
        let catalog = KernelCatalog::load(&[path]).unwrap();
        request.grid = [0, 1, 1];
        assert!(catalog.select("pascal.f32.scalar", &request).is_err());
        request.grid = [1, 1, 1];
        request.max_dynamic_shared_mem_bytes = 0;
        assert!(catalog.select("pascal.f32.scalar", &request).is_ok());
        request.args[0] = Argument::Tensor {
            dtype: ElementType::F32,
            shape: vec![usize::MAX, 3],
            contiguous: true,
            start_offset: 0,
            representation: None,
        };
        assert!(catalog.select("pascal.f32.scalar", &request).is_err());
    }

    #[test]
    fn discovers_external_packages_and_selects_by_device_capability() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("kernel-packages/cuda");
        let catalog = KernelCatalog::discover(&root).unwrap();
        assert_eq!(catalog.len(), 2);

        let tensor = Argument::Tensor {
            dtype: ElementType::F32,
            shape: vec![256],
            contiguous: true,
            start_offset: 0,
            representation: None,
        };
        let mut request = LaunchRequest {
            operation: "tensor.elementwise.scale.v1".into(),
            cc: 61,
            args: vec![
                tensor.clone(),
                Argument::Scalar {
                    dtype: ElementType::F32,
                },
                tensor,
                Argument::Scalar {
                    dtype: ElementType::U32,
                },
            ],
            grid: [2, 1, 1],
            max_dynamic_shared_mem_bytes: 0,
        };
        let low = catalog.resolve(&request).unwrap();
        assert_eq!(low.manifest.target_cc, 61);
        request.cc = 80;
        let high = catalog.resolve(&request).unwrap();
        assert_eq!(high.manifest.target_cc, 80);
        request.cc = 90;
        assert_eq!(catalog.resolve(&request).unwrap().manifest.target_cc, 80);
        assert_eq!(
            low.ptx.replace(".target sm_61", ".target sm_generic"),
            high.ptx.replace(".target sm_80", ".target sm_generic")
        );
        request.cc = 60;
        assert!(catalog.resolve(&request).is_err());
    }

    #[test]
    fn duplicate_architecture_or_ambiguous_provider_is_refused() {
        let (_dir, path, request) = fixture();
        assert!(KernelCatalog::load(&[path.clone(), path.clone()]).is_err());
        let (other_dir, other_path, _) = fixture();
        let mut other: KernelManifest =
            serde_json::from_slice(&std::fs::read(&other_path).unwrap()).unwrap();
        other.implementation_id = "alternative".into();
        std::fs::write(&other_path, serde_json::to_vec(&other).unwrap()).unwrap();
        let catalog = KernelCatalog::load(&[path, other_dir.path().join("manifest.json")]).unwrap();
        assert!(catalog.resolve(&request).is_err());
        assert!(catalog.select("pascal.f32.scalar", &request).is_ok());
    }

    #[test]
    fn refuses_path_escape_and_wrong_gpu() {
        let (_dir, path, _) = fixture();
        let mut manifest: KernelManifest =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest.artifact = PathBuf::from("../escape.ptx");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(VerifiedKernel::read(&path).is_err());
        let (_dir, path, mut request) = fixture();
        request.cc = 80;
        assert!(KernelCatalog::load(&[path])
            .unwrap()
            .select("pascal.f32.scalar", &request)
            .is_err());
    }
}
