use crate::{digest, VerifiedModule, ABI};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Component, Path},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub const PROFILE_SCHEMA: &str = "ASD-LINEAR-PROFILE-V1";
pub const PROFILE_ENV: &str = "MISTRALRS_ASD_F32_PROFILE";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub id: String,
    pub shape: [usize; 3],
    pub implementation_id: String,
    pub manifest: String,
    pub manifest_sha256: String,
    pub artifact_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExactProfile {
    pub schema: String,
    pub profile_id: String,
    pub abi: String,
    pub purpose: String,
    pub target_cc: u32,
    pub decisions: Vec<Decision>,
}

pub struct BoundDecision {
    pub decision: Decision,
    pub identity: String,
    pub module: Arc<VerifiedModule>,
    pub submissions: AtomicU64,
}

pub struct LoadedProfile {
    pub profile: ExactProfile,
    pub sha256: String,
    pub decisions: Vec<Arc<BoundDecision>>,
}

impl LoadedProfile {
    pub fn read(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let dir = path.parent().context("profile has no directory")?;
        let bytes = std::fs::read(&path)?;
        let profile: ExactProfile = serde_json::from_slice(&bytes)?;
        ensure!(
            profile.schema == PROFILE_SCHEMA && profile.abi == ABI,
            "unsupported linear profile schema/ABI"
        );
        ensure!(
            profile.purpose == "validation_only",
            "only validation profiles are supported; no performance promotion"
        );
        ensure!(
            !profile.profile_id.is_empty() && !profile.decisions.is_empty(),
            "empty linear profile"
        );
        let mut keys = HashSet::new();
        let mut ids = HashSet::new();
        let mut modules = HashMap::<String, (String, Arc<VerifiedModule>)>::new();
        let mut decisions = Vec::new();
        for row in &profile.decisions {
            ensure!(
                !row.id.is_empty() && ids.insert(&row.id) && keys.insert(row.shape),
                "duplicate or empty exact decision"
            );
            let name = Path::new(&row.manifest);
            ensure!(
                name.components().all(|p| matches!(p, Component::Normal(_))),
                "manifest must be relative to profile"
            );
            let manifest_path = dir.join(name).canonicalize()?;
            ensure!(
                manifest_path.starts_with(dir),
                "manifest link escapes profile directory"
            );
            let module = if let Some((hash, module)) = modules.get(&row.manifest) {
                ensure!(
                    hash == &row.manifest_sha256,
                    "conflicting manifest hash for shared module"
                );
                module.clone()
            } else {
                let manifest_bytes = std::fs::read(&manifest_path)?;
                ensure!(
                    digest(&manifest_bytes) == row.manifest_sha256,
                    "profile manifest hash mismatch"
                );
                let module = Arc::new(VerifiedModule::read_bytes(&manifest_path, &manifest_bytes)?);
                modules.insert(
                    row.manifest.clone(),
                    (row.manifest_sha256.clone(), module.clone()),
                );
                module
            };
            ensure!(
                module.manifest.implementation_id == row.implementation_id
                    && module.manifest.sha256 == row.artifact_sha256,
                "profile/artifact identity mismatch"
            );
            module.manifest.admit(profile.target_cc, row.shape)?;
            let identity = digest(&serde_json::to_vec(&(
                PROFILE_SCHEMA,
                &profile.profile_id,
                row,
            ))?);
            decisions.push(Arc::new(BoundDecision {
                decision: row.clone(),
                identity,
                module,
                submissions: AtomicU64::new(0),
            }));
        }
        Ok(Self {
            profile,
            sha256: digest(&bytes),
            decisions,
        })
    }

    pub fn evidence(&self) -> serde_json::Value {
        serde_json::json!({"schema":PROFILE_SCHEMA, "profile_id":self.profile.profile_id,
            "profile_sha256":self.sha256, "purpose":self.profile.purpose,
            "decisions":self.decisions.iter().map(|d| serde_json::json!({
                "decision_id":d.decision.id, "shape":d.decision.shape,
                "decision_identity_sha256":d.identity, "implementation_id":d.decision.implementation_id,
                "artifact_sha256":d.decision.artifact_sha256, "entry":d.module.manifest.entry,
                "submissions":d.submissions.load(Ordering::Relaxed)
            })).collect::<Vec<_>>()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;

    fn package() -> (tempfile::TempDir, std::path::PathBuf, ExactProfile) {
        let dir = tempfile::tempdir().unwrap();
        let ptx = b".version 8.8\n.target sm_61\n.address_size 64\n.visible .entry linear() {}\n";
        let artifact_hash = digest(ptx);
        let manifest = Manifest {
            schema: 1,
            abi: ABI.into(),
            implementation_id: "opaque.provider".into(),
            artifact: "kernel.ptx".into(),
            sha256: artifact_hash.clone(),
            entry: "linear".into(),
            target_cc: 61,
            min_cc: 61,
            max_cc: 61,
            dtype: "f32".into(),
            layout: "contiguous_zero_offset".into(),
            bias: false,
            block: [16, 16, 1],
            max_shape: [4096; 3],
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(dir.path().join("module.json"), &bytes).unwrap();
        std::fs::write(dir.path().join("kernel.ptx"), ptx).unwrap();
        let profile = ExactProfile {
            schema: PROFILE_SCHEMA.into(),
            profile_id: "manual.validation".into(),
            abi: ABI.into(),
            purpose: "validation_only".into(),
            target_cc: 61,
            decisions: vec![Decision {
                id: "prefill".into(),
                shape: [3, 17, 33],
                implementation_id: manifest.implementation_id,
                manifest: "module.json".into(),
                manifest_sha256: digest(&bytes),
                artifact_sha256: artifact_hash,
            }],
        };
        let path = dir.path().join("profile.json");
        std::fs::write(&path, serde_json::to_vec(&profile).unwrap()).unwrap();
        (dir, path, profile)
    }

    fn rejects(path: &Path, profile: &ExactProfile, reason: &str) {
        std::fs::write(path, serde_json::to_vec(profile).unwrap()).unwrap();
        let error = LoadedProfile::read(path).err().expect("expected rejection");
        assert!(
            error.to_string().contains(reason),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn exact_profile_is_an_immutable_cold_snapshot_without_cuda() {
        let (dir, path, _) = package();
        let loaded = LoadedProfile::read(&path).unwrap();
        let identity = loaded.decisions[0].identity.clone();
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(dir.path().join("module.json")).unwrap();
        std::fs::remove_file(dir.path().join("kernel.ptx")).unwrap();
        assert_eq!(
            loaded.evidence()["decisions"][0]["decision_identity_sha256"],
            identity
        );
        assert_eq!(loaded.decisions[0].decision.shape, [3, 17, 33]);
    }

    #[test]
    fn rejects_duplicate_exact_keys_and_decision_ids() {
        let (_dir, path, mut p) = package();
        let mut duplicate = p.decisions[0].clone();
        duplicate.id = "other".into();
        p.decisions.push(duplicate);
        rejects(&path, &p, "duplicate");
        p.decisions[1].shape[0] = 1;
        p.decisions[1].id = p.decisions[0].id.clone();
        rejects(&path, &p, "duplicate");
    }

    #[test]
    fn rejects_identity_changes_and_conflicting_shared_manifest_hashes() {
        let (_dir, path, p) = package();
        let mut invalid = p.clone();
        invalid.decisions[0].artifact_sha256 = "0".repeat(64);
        rejects(&path, &invalid, "identity mismatch");
        let mut invalid = p.clone();
        invalid.decisions[0].implementation_id = "wrong".into();
        rejects(&path, &invalid, "identity mismatch");
        let mut invalid = p.clone();
        invalid.decisions[0].manifest_sha256 = "0".repeat(64);
        rejects(&path, &invalid, "manifest hash mismatch");
        let mut invalid = p.clone();
        let mut second = p.decisions[0].clone();
        second.id = "decode".into();
        second.shape[0] = 1;
        second.manifest_sha256 = "0".repeat(64);
        invalid.decisions.push(second);
        rejects(&path, &invalid, "conflicting manifest hash");
    }

    #[test]
    fn rejects_promotion_unknown_abi_architecture_and_escaped_paths() {
        let (_dir, path, p) = package();
        let mut invalid = p.clone();
        invalid.purpose = "promoted".into();
        rejects(&path, &invalid, "only validation profiles");
        let mut invalid = p.clone();
        invalid.abi = "asd.xwo.f32.v1".into();
        rejects(&path, &invalid, "unsupported linear profile");
        let mut invalid = p.clone();
        invalid.target_cc = 75;
        rejects(&path, &invalid, "architecture outside");
        let mut invalid = p.clone();
        invalid.decisions[0].manifest = "../module.json".into();
        rejects(&path, &invalid, "manifest must be relative");
    }

    #[test]
    fn decision_identity_changes_with_shape_and_artifact() {
        let (_dir, path, mut p) = package();
        let before = LoadedProfile::read(&path).unwrap().decisions[0]
            .identity
            .clone();
        p.decisions[0].shape[0] = 1;
        std::fs::write(&path, serde_json::to_vec(&p).unwrap()).unwrap();
        let after = LoadedProfile::read(&path).unwrap().decisions[0]
            .identity
            .clone();
        assert_ne!(before, after);
    }
}
