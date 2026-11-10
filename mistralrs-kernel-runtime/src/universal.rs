use crate::digest;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

pub const TYPED_ABI: &str = "dkr.typed.v1";
pub const MODULE_SCHEMA: &str = "DKR-KERNEL-MODULE-V1";
pub const MAX_MANIFEST_BYTES: u64 = 256 * 1024;
pub const MAX_KERNEL_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_ARGUMENTS: usize = 32;
pub const MAX_TENSOR_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
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
            Self::I64 | Self::U64 => 8,
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

#[derive(Clone, Debug)]
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
    pub ptx: String,
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
            "DKR-R1 accepts only relative PTX artifacts"
        );
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

impl VerifiedKernel {
    pub fn read(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        ensure!(
            std::fs::metadata(&path)?.len() <= MAX_MANIFEST_BYTES,
            "manifest budget exceeded"
        );
        let bytes = std::fs::read(&path)?;
        let manifest: KernelManifest = serde_json::from_slice(&bytes)?;
        manifest.validate()?;
        let root = path.parent().context("manifest directory missing")?;
        let artifact_path = root.join(&manifest.artifact).canonicalize()?;
        ensure!(
            artifact_path.starts_with(root),
            "kernel artifact escapes package directory"
        );
        ensure!(
            std::fs::metadata(&artifact_path)?.len() <= MAX_KERNEL_BYTES,
            "kernel artifact budget exceeded"
        );
        let artifact = std::fs::read(artifact_path)?;
        ensure!(
            digest(&artifact).eq_ignore_ascii_case(&manifest.sha256),
            "kernel artifact digest mismatch"
        );
        let ptx = String::from_utf8(artifact)?;
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
        Ok(Self {
            manifest,
            ptx,
            manifest_sha256: digest(&bytes),
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
    modules: HashMap<(String, String), Arc<VerifiedKernel>>,
}

impl KernelCatalog {
    pub fn load(paths: &[PathBuf]) -> Result<Self> {
        let mut result = Self::default();
        for path in paths {
            let module = Arc::new(VerifiedKernel::read(path)?);
            let key = (
                module.manifest.operation.clone(),
                module.manifest.implementation_id.clone(),
            );
            ensure!(
                !result.modules.contains_key(&key),
                "duplicate operation/implementation identity"
            );
            result.modules.insert(key, module);
        }
        Ok(result)
    }

    pub fn select(
        &self,
        implementation_id: &str,
        request: &LaunchRequest,
    ) -> Result<&Arc<VerifiedKernel>> {
        let key = (request.operation.clone(), implementation_id.to_owned());
        let module = self
            .modules
            .get(&key)
            .context("no explicitly selected kernel provider")?;
        module.manifest.admit(request)?;
        Ok(module)
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
