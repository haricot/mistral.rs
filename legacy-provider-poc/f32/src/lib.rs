use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Layout};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path};

pub const ABI: &str = "mistral.linear.f32.v1";
pub const MAX_ARTIFACT_BYTES: u64 = 8 * 1024 * 1024;
pub const MAX_TENSOR_BYTES: usize = 64 * 1024 * 1024;
pub const ATOL: f32 = 1e-5;
pub const RTOL: f32 = 1e-5;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub abi: String,
    pub implementation_id: String,
    pub artifact: String,
    pub sha256: String,
    pub entry: String,
    pub target_cc: u32,
    pub min_cc: u32,
    pub max_cc: u32,
    pub dtype: String,
    pub layout: String,
    pub bias: bool,
    pub block: [u32; 3],
    pub max_shape: [usize; 3],
}

pub struct VerifiedModule {
    pub manifest: Manifest,
    pub ptx: String,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Manifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == 1 && self.abi == ABI,
            "unsupported ABI/schema"
        );
        ensure!(
            !self.implementation_id.is_empty(),
            "empty implementation identity"
        );
        ensure!(
            self.dtype == "f32" && self.layout == "contiguous_zero_offset" && !self.bias,
            "ABI requires F32, contiguous zero-offset tensors and no bias"
        );
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid SHA256"
        );
        ensure!(
            !self.entry.is_empty()
                && self
                    .entry
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_'),
            "invalid entry symbol"
        );
        ensure!(
            self.target_cc >= 50 && self.min_cc >= self.target_cc && self.max_cc >= self.min_cc,
            "invalid architecture bounds"
        );
        let [bx, by, bz] = self.block;
        ensure!(
            bx > 0 && by > 0 && bz == 1 && u64::from(bx) * u64::from(by) <= 1024,
            "invalid output-thread launch block"
        );
        ensure!(
            self.max_shape
                .iter()
                .all(|&v| v > 0 && v <= i32::MAX as usize),
            "invalid shape bounds"
        );
        let path = Path::new(&self.artifact);
        ensure!(
            !path.as_os_str().is_empty()
                && path.components().all(|c| matches!(c, Component::Normal(_)))
                && path.extension().is_some_and(|s| s == "ptx"),
            "artifact must be a relative PTX path"
        );
        Ok(())
    }

    pub fn admit(&self, cc: u32, shape: [usize; 3]) -> Result<()> {
        self.validate()?;
        ensure!(
            (self.min_cc..=self.max_cc).contains(&cc),
            "architecture outside provider contract"
        );
        ensure!(
            shape
                .iter()
                .zip(self.max_shape)
                .all(|(&v, max)| v > 0 && v <= max),
            "shape outside provider contract"
        );
        let [m, n, k] = shape;
        let elements = m.checked_mul(k).and_then(|a| {
            n.checked_mul(k).and_then(|b| {
                m.checked_mul(n)
                    .and_then(|c| a.checked_add(b)?.checked_add(c))
            })
        });
        let bytes = elements.and_then(|e| e.checked_mul(std::mem::size_of::<f32>()));
        ensure!(
            bytes.is_some_and(|b| b <= MAX_TENSOR_BYTES),
            "tensor resource budget exceeded"
        );
        Ok(())
    }

    pub fn module_key(&self) -> String {
        format!("{ABI}:{}", self.sha256.to_ascii_lowercase())
    }
}

impl VerifiedModule {
    pub fn read(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let dir = path.parent().context("manifest has no directory")?;
        let manifest: Manifest = serde_json::from_reader(std::fs::File::open(&path)?)?;
        manifest.validate()?;
        let artifact = dir.join(&manifest.artifact).canonicalize()?;
        ensure!(
            artifact.starts_with(dir),
            "artifact link escapes package directory"
        );
        ensure!(
            std::fs::metadata(&artifact)?.len() <= MAX_ARTIFACT_BYTES,
            "artifact budget exceeded"
        );
        let bytes = std::fs::read(&artifact)?;
        ensure!(
            bytes.len() as u64 <= MAX_ARTIFACT_BYTES,
            "artifact budget exceeded"
        );
        ensure!(
            digest(&bytes).eq_ignore_ascii_case(&manifest.sha256),
            "artifact SHA256 mismatch"
        );
        let ptx = String::from_utf8(bytes)?;
        let tokens: Vec<_> = ptx.split_whitespace().collect();
        ensure!(
            tokens.windows(2).any(|t| t == [".address_size", "64"]),
            "ABI requires 64-bit PTX pointers"
        );
        let target = format!("sm_{}", manifest.target_cc);
        ensure!(
            tokens
                .windows(2)
                .any(|t| t[0] == ".target" && t[1].trim_end_matches(',') == target),
            "PTX target differs from manifest"
        );
        ensure!(
            tokens.windows(2).any(
                |t| t[0] == ".entry" && t[1].split('(').next() == Some(manifest.entry.as_str())
            ),
            "entry symbol missing from PTX"
        );
        Ok(Self { manifest, ptx })
    }
}

pub fn tensor_contract(dtype: DType, layout: &Layout) -> Result<(usize, usize)> {
    ensure!(dtype == DType::F32, "ABI requires F32");
    ensure!(
        layout.is_contiguous() && layout.start_offset() == 0,
        "ABI requires contiguous zero-offset tensors"
    );
    Ok(layout.shape().dims2()?)
}

pub fn oracle(x: &[f32], w: &[f32], [m, n, k]: [usize; 3]) -> Vec<f32> {
    (0..m * n)
        .map(|idx| {
            let row = idx / n;
            let col = idx % n;
            (0..k)
                .map(|i| f64::from(x[row * k + i]) * f64::from(w[col * k + i]))
                .sum::<f64>() as f32
        })
        .collect()
}

pub fn compare(actual: &[f32], expected: &[f32]) -> Result<f32> {
    ensure!(
        actual.len() == expected.len() && !actual.is_empty(),
        "wrong output length"
    );
    let mut max_error = 0f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let error = (a - b).abs();
        if !a.is_finite() || !b.is_finite() || error > ATOL + RTOL * b.abs() {
            bail!("numerical mismatch at {index}: actual={a}, expected={b}");
        }
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn manifest() -> Manifest {
        Manifest {
            schema: 1,
            abi: ABI.into(),
            implementation_id: "replaceable".into(),
            artifact: "kernel.ptx".into(),
            sha256: "0".repeat(64),
            entry: "linear".into(),
            target_cc: 61,
            min_cc: 61,
            max_cc: 61,
            dtype: "f32".into(),
            layout: "contiguous_zero_offset".into(),
            bias: false,
            block: [16, 16, 1],
            max_shape: [4096; 3],
        }
    }

    fn fixture() -> (tempfile::TempDir, std::path::PathBuf, Manifest) {
        let dir = tempfile::tempdir().unwrap();
        let ptx = b".version 8.8\n.target sm_61\n.address_size 64\n.visible .entry linear() {}\n";
        let mut m = manifest();
        m.sha256 = digest(ptx);
        std::fs::write(dir.path().join(&m.artifact), ptx).unwrap();
        let path = dir.path().join("module.json");
        std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
        (dir, path, m)
    }

    #[test]
    fn validates_package_without_cuda() {
        let (_dir, path, _) = fixture();
        let module = VerifiedModule::read(&path).unwrap();
        module.manifest.admit(61, [5, 17, 33]).unwrap();
    }

    #[test]
    fn rejects_artifact_changes_and_missing_symbol_or_wrong_target() {
        let (dir, path, mut m) = fixture();
        std::fs::write(dir.path().join(&m.artifact), "tampered").unwrap();
        assert!(VerifiedModule::read(&path)
            .err()
            .unwrap()
            .to_string()
            .contains("SHA256"));
        for (ptx, reason) in [
            (
                ".target sm_61\n.address_size 64\n.entry other() {}",
                "entry symbol missing",
            ),
            (
                ".target sm_75\n.address_size 64\n.entry linear() {}",
                "PTX target differs",
            ),
            (
                ".target sm_61\n.address_size 32\n.entry linear() {}",
                "64-bit PTX pointers",
            ),
        ] {
            m.sha256 = digest(ptx.as_bytes());
            std::fs::write(dir.path().join(&m.artifact), ptx).unwrap();
            std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
            assert!(VerifiedModule::read(&path)
                .err()
                .unwrap()
                .to_string()
                .contains(reason));
        }
    }

    #[test]
    fn rejects_oversized_artifact_before_reading() {
        let (dir, path, m) = fixture();
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join(m.artifact))
            .unwrap()
            .set_len(MAX_ARTIFACT_BYTES + 1)
            .unwrap();
        assert!(VerifiedModule::read(&path)
            .err()
            .unwrap()
            .to_string()
            .contains("artifact budget"));
    }

    #[test]
    fn rejects_incompatible_contracts_and_budgets() {
        let m = manifest();
        assert!(m.admit(75, [1, 17, 33]).is_err());
        for shape in [[0, 17, 33], [4097, 17, 33], [4096, 4096, 4096]] {
            assert!(m.admit(61, shape).is_err());
        }
        for field in ["abi", "dtype", "layout", "artifact", "sha256"] {
            let mut value = serde_json::to_value(&m).unwrap();
            value[field] = "invalid".into();
            assert!(serde_json::from_value::<Manifest>(value)
                .unwrap()
                .validate()
                .is_err());
        }
        for block in [[0, 16, 1], [1024, 2, 1], [16, 16, 2]] {
            let mut invalid = m.clone();
            invalid.block = block;
            assert!(invalid.validate().is_err());
        }
        let mut invalid = m.clone();
        invalid.bias = true;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn rejects_offset_strides_dtype_and_rank() -> Result<()> {
        let x = Tensor::zeros((4, 33), DType::F32, &Device::Cpu)?;
        tensor_contract(x.dtype(), x.layout())?;
        for bad in [x.narrow(0, 1, 2)?, x.t()?, x.flatten_all()?] {
            assert!(tensor_contract(bad.dtype(), bad.layout()).is_err());
        }
        assert!(tensor_contract(DType::F16, x.layout()).is_err());
        Ok(())
    }

    #[test]
    fn rejects_unknown_manifest_fields_and_invalid_entry() {
        let mut value = serde_json::to_value(manifest()).unwrap();
        value["extra"] = true.into();
        assert!(serde_json::from_value::<Manifest>(value).is_err());
        let mut invalid = manifest();
        invalid.entry = "bad-symbol".into();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn cache_identity_follows_content_even_for_same_provider_name() {
        let a = manifest();
        let mut b = a.clone();
        b.sha256 = "1".repeat(64);
        assert_eq!(a.implementation_id, b.implementation_id);
        assert_ne!(a.module_key(), b.module_key());
        assert!(compare(&[f32::NAN], &[0.]).is_err());
        assert!(compare(&[1.], &[2.]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_artifact_symlink_outside_package() {
        let (dir, path, m) = fixture();
        let external = tempfile::NamedTempFile::new().unwrap();
        std::fs::remove_file(dir.path().join(&m.artifact)).unwrap();
        std::os::unix::fs::symlink(external.path(), dir.path().join(&m.artifact)).unwrap();
        assert!(VerifiedModule::read(&path).is_err());
    }

    #[test]
    fn independent_oracle_matches_candle_cpu() -> Result<()> {
        let (m, n, k) = (5, 17, 33);
        let x: Vec<_> = (0..m * k).map(|i| (i as f32 % 13. - 6.) / 8.).collect();
        let w: Vec<_> = (0..n * k).map(|i| (i as f32 % 17. - 8.) / 16.).collect();
        let expected = oracle(&x, &w, [m, n, k]);
        let actual = Tensor::from_vec(x, (m, k), &Device::Cpu)?
            .matmul(&Tensor::from_vec(w, (n, k), &Device::Cpu)?.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        compare(&actual, &expected)?;
        Ok(())
    }
}
