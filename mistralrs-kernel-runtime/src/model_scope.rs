//! Model-neutral capability requirements and immutable verified catalogue identity.
//! A descriptor is an explicit declaration by a producer, not proof of complete
//! model reachability. No architecture names or GPU-specific kernel policy here.
use crate::{digest, universal::{KernelCatalog, KernelManifest, Parameter, MAX_CATALOG_PROFILES}};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    io::Read,
    path::Path,
    sync::Mutex,
};

pub const CAPABILITY_SCHEMA: &str = "DKR-MODEL-CAPABILITIES-V1";
pub const SELECTION_SCHEMA: &str = "DKR-MODEL-SELECTION-V1";
pub const SCOPE_SCHEMA: &str = "DKR-ACTIVE-SCOPE-V1";
pub const SCOPE_FILENAME: &str = "dkr-active-scope.json";
const MAX_RECEIPT_BYTES: u64 = 16 * 1024;
const MAX_MASTER_PROFILES: usize = 262144;

fn identifier(s: &str) -> bool {
    !s.is_empty() && s.len() <= 512 &&
        s.bytes().all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
}
fn sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileIdentity {
    pub operation: String,
    pub implementation_id: String,
    pub target_cc: u32,
}
impl ProfileIdentity {
    fn validate(&self) -> Result<()> {
        ensure!(identifier(&self.operation) && identifier(&self.implementation_id)
            && (50..=255).contains(&self.target_cc), "invalid profile identity");
        Ok(())
    }
}
impl From<&KernelManifest> for ProfileIdentity {
    fn from(m: &KernelManifest) -> Self {
        Self { operation: m.operation.clone(), implementation_id: m.implementation_id.clone(), target_cc: m.target_cc }
    }
}

/// None retains every device-compatible implementation of the operation.
/// Some lists exact implementation IDs, never a wildcard or a guessed dtype.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationRequirement {
    pub operation: String,
    pub implementations: Option<Vec<String>>,
}
impl OperationRequirement {
    fn validate(&self) -> Result<()> {
        ensure!(identifier(&self.operation), "invalid operation requirement");
        if let Some(ids) = &self.implementations {
            ensure!(!ids.is_empty() && ids.len() <= MAX_CATALOG_PROFILES
                && ids.iter().all(|x| identifier(x))
                && ids.iter().collect::<BTreeSet<_>>().len() == ids.len(),
                "invalid implementation requirement");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityComponent {
    pub id: String,
    pub role: String,
    pub metadata_sha256: String,
    pub facts: BTreeMap<String, Value>,
    pub requirements: Vec<OperationRequirement>,
    pub unresolved: Vec<String>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coverage { Partial, DeclaredComplete }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilityDescriptor {
    pub schema: String,
    pub model: String,
    pub device_cc: u32,
    pub catalog_revision: String,
    pub components: Vec<CapabilityComponent>,
    pub request_modes: Vec<String>,
    pub request_requirements: Vec<OperationRequirement>,
    pub unresolved: Vec<String>,
    pub coverage: Coverage,
}
impl ModelCapabilityDescriptor {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == CAPABILITY_SCHEMA && !self.model.trim().is_empty()
            && !self.catalog_revision.trim().is_empty() && (50..=255).contains(&self.device_cc),
            "invalid descriptor header");
        ensure!(!self.components.is_empty() && self.components.len() <= MAX_CATALOG_PROFILES,
            "invalid component count");
        let mut ids = BTreeSet::new();
        for c in &self.components {
            ensure!(identifier(&c.id) && identifier(&c.role) && ids.insert(&c.id)
                && sha256(&c.metadata_sha256), "invalid or duplicate capability component");
            for r in &c.requirements { r.validate()?; }
        }
        ensure!(self.components.iter().any(|c| c.role == "target"), "missing target component");
        ensure!(self.request_modes.iter().all(|m| identifier(m))
            && self.request_modes.iter().collect::<BTreeSet<_>>().len() == self.request_modes.len(),
            "invalid request modes");
        for r in &self.request_requirements { r.validate()?; }
        if self.coverage == Coverage::DeclaredComplete {
            ensure!(!self.request_modes.is_empty() && self.unresolved.is_empty()
                && self.components.iter().all(|c| c.unresolved.is_empty()),
                "declared complete descriptor has unresolved capabilities or request policy");
        }
        Ok(())
    }
}

/// Metadata-only view of the offline master inventory. Never admits an image.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileMetadata {
    pub operation: String,
    pub implementation_id: String,
    pub target_cc: u32,
    pub min_cc: u32,
    pub max_cc: u32,
}
impl ProfileMetadata {
    fn identity(&self) -> ProfileIdentity {
        ProfileIdentity { operation: self.operation.clone(), implementation_id: self.implementation_id.clone(), target_cc: self.target_cc }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileDependency {
    pub profile: ProfileIdentity,
    pub requires: Vec<OperationRequirement>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSelection {
    pub schema: String,
    pub model: String,
    pub device_cc: u32,
    pub profiles: Vec<ProfileIdentity>,
}
#[derive(Clone, Debug)]
pub struct ScopePlan {
    pub selection: ModelSelection,
    /// Producer's declaration only, not qualified reachability.
    pub coverage_declared: bool,
}
impl ScopePlan {
    pub fn materialization_selection(&self) -> Result<&ModelSelection> {
        ensure!(self.coverage_declared, "partial descriptor cannot produce a deployment selection");
        Ok(&self.selection)
    }
}

/// Union all components and request modes, then recursively resolve explicit
/// profile dependencies. An omitted implementation filter keeps all compatible
/// variants. No implicit fallback, no dispatch-time filtering.
pub fn resolve_model_scope(
    descriptor: &ModelCapabilityDescriptor,
    master: &[ProfileMetadata],
    dependencies: &[ProfileDependency],
) -> Result<ScopePlan> {
    descriptor.validate()?;
    ensure!(!master.is_empty() && master.len() <= MAX_MASTER_PROFILES, "master metadata bound");
    let mut identities = BTreeMap::new();
    let mut operations: BTreeMap<&str, Vec<&ProfileMetadata>> = BTreeMap::new();
    for p in master {
        let id = p.identity();
        id.validate()?;
        ensure!(p.target_cc <= p.min_cc && p.min_cc <= p.max_cc,
            "invalid profile architecture range");
        ensure!(identities.insert(id, p).is_none(), "duplicate master identity");
        if (p.min_cc..=p.max_cc).contains(&descriptor.device_cc) {
            operations.entry(&p.operation).or_default().push(p);
        }
    }
    let mut deps = BTreeMap::new();
    for rule in dependencies {
        ensure!(identities.contains_key(&rule.profile), "dependency owner missing from master");
        for r in &rule.requires { r.validate()?; }
        ensure!(deps.insert(rule.profile.clone(), &rule.requires).is_none(), "duplicate dependency owner");
    }
    let mut queue: VecDeque<&OperationRequirement> = descriptor.components.iter()
        .flat_map(|c| c.requirements.iter()).chain(descriptor.request_requirements.iter()).collect();
    let mut selected = BTreeSet::new();
    while let Some(r) = queue.pop_front() {
        let candidates = operations.get(r.operation.as_str())
            .with_context(|| format!("no compatible profile for {}", r.operation))?;
        let mut matched = BTreeSet::new();
        for p in candidates {
            if r.implementations.as_ref().is_some_and(|ids| !ids.contains(&p.implementation_id)) {
                continue;
            }
            matched.insert(p.implementation_id.as_str());
            let id = p.identity();
            if selected.insert(id.clone()) {
                ensure!(selected.len() <= MAX_CATALOG_PROFILES, "active logical profile bound");
                if let Some(next) = deps.get(&id) { queue.extend(next.iter()); }
            }
        }
        ensure!(!matched.is_empty(), "requirement selected no compatible profile");
        if let Some(ids) = &r.implementations {
            ensure!(ids.iter().all(|id| matched.contains(id.as_str())), "required implementation unavailable");
        }
    }
    ensure!(!selected.is_empty(), "empty active selection");
    Ok(ScopePlan {
        selection: ModelSelection { schema: SELECTION_SCHEMA.into(), model: descriptor.model.clone(),
            device_cc: descriptor.device_cc, profiles: selected.into_iter().collect() },
        coverage_declared: descriptor.coverage == Coverage::DeclaredComplete,
    })
}

/// Canonical array encoding shared with the offline packager. All semantic
/// manifest fields participate, including optional CUBIN and representations.
/// Original V1 bytes / per-profile hashes remain separate module-cache identities.
fn manifest_row(m: &KernelManifest) -> Value {
    let args: Vec<Value> = m.arguments.iter().map(|p| match p {
        Parameter::Tensor {name,dtype,rank,access,representation} =>
            json!(["tensor",name,dtype,rank,access,representation]),
        Parameter::Scalar {name,dtype} => json!(["scalar",name,dtype]),
    }).collect();
    json!([m.schema,m.abi,m.operation,m.implementation_id,m.entry,m.artifact,
        m.sha256.to_ascii_lowercase(),m.cubin_artifact,
        m.cubin_sha256.as_ref().map(|s|s.to_ascii_lowercase()),
        m.target_cc,m.min_cc,m.max_cc,m.block,m.shared_mem_bytes,args])
}
pub fn verified_scope_id(catalog: &KernelCatalog, cc: u32) -> Result<String> {
    ensure!((50..=255).contains(&cc) && !catalog.is_empty(), "invalid scope device or empty catalogue");
    let mut modules: Vec<_> = catalog.verified_modules().collect();
    modules.sort_by(|a,b| {
        let a=&a.manifest; let b=&b.manifest;
        (&a.operation,&a.implementation_id,a.target_cc).cmp(&(&b.operation,&b.implementation_id,b.target_cc))
    });
    let mut rows = Vec::with_capacity(modules.len());
    for m in modules {
        ensure!((m.manifest.min_cc..=m.manifest.max_cc).contains(&cc), "scope contains incompatible profile");
        rows.push(manifest_row(&m.manifest));
    }
    Ok(digest(&serde_json::to_vec(&json!([SCOPE_SCHEMA,cc,rows]))?))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeReceipt {
    pub schema: String,
    pub device_cc: u32,
    pub scope_id: String,
    pub profile_count: usize,
}
impl ScopeReceipt {
    pub fn verify(&self, catalog: &KernelCatalog) -> Result<()> {
        ensure!(self.schema == SCOPE_SCHEMA && sha256(&self.scope_id)
            && self.profile_count == catalog.len(), "invalid scope receipt");
        ensure!(verified_scope_id(catalog, self.device_cc)? == self.scope_id, "scope content identity mismatch");
        Ok(())
    }
}
pub fn read_scope_receipt(root: &Path) -> Result<Option<ScopeReceipt>> {
    let path = root.join(SCOPE_FILENAME);
    match std::fs::symlink_metadata(&path) {
        Ok(meta) => ensure!(meta.is_file() && !meta.file_type().is_symlink()
            && meta.len() <= MAX_RECEIPT_BYTES, "invalid scope receipt file"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let mut data = Vec::new();
    std::fs::File::open(path)?.take(MAX_RECEIPT_BYTES + 1).read_to_end(&mut data)?;
    ensure!(data.len() as u64 <= MAX_RECEIPT_BYTES, "scope receipt budget");
    Ok(Some(serde_json::from_slice(&data)?))
}

/// Process-global callers cannot switch the catalogue underneath their
/// OnceLock/graph/launch-plan caches. Unscoped legacy callers keep their old
/// behavior, but cannot transition into a scoped context in that process.
#[derive(Default)]
pub struct ScopeLatch {
    state: Mutex<Option<Option<ScopeReceipt>>>,
}
impl ScopeLatch {
    pub fn admit(&self, receipt: Option<&ScopeReceipt>) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| anyhow::anyhow!("scope latch poisoned"))?;
        match &*state {
            None => *state = Some(receipt.cloned()),
            Some(previous) if previous.as_ref() == receipt => (),
            _ => bail!("DKR active scope differs from process snapshot; use a union or isolated process"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn req(op:&str,ids:Option<&[&str]>)->OperationRequirement{
        OperationRequirement{operation:op.into(),implementations:ids.map(|x|x.iter().map(|v|v.to_string()).collect())}
    }
    fn profile(op:&str,imp:&str,cc:u32)->ProfileMetadata{
        ProfileMetadata{operation:op.into(),implementation_id:imp.into(),target_cc:cc,min_cc:cc,max_cc:cc}
    }
    fn descriptor()->ModelCapabilityDescriptor{
        ModelCapabilityDescriptor{schema:CAPABILITY_SCHEMA.into(),model:"synthetic".into(),device_cc:61,
            catalog_revision:"fixture".into(),components:vec![
                CapabilityComponent{id:"target".into(),role:"target".into(),metadata_sha256:"0".repeat(64),
                    facts:BTreeMap::new(),requirements:vec![req("router",Some(&["e64"]))],unresolved:vec![]},
                CapabilityComponent{id:"draft".into(),role:"draft".into(),metadata_sha256:"1".repeat(64),
                    facts:BTreeMap::new(),requirements:vec![req("router",Some(&["e128"]))],unresolved:vec![]}
            ],request_modes:vec!["greedy".into(),"topk".into()],request_requirements:vec![req("sampling",None)],
            unresolved:vec![],coverage:Coverage::DeclaredComplete}
    }
    #[test]
    fn dkr_model_scope_resolves_union_and_dependency_cycle(){
        let master=[profile("router","e64",61),profile("router","e128",61),profile("router","e256",61),
            profile("sampling","a",61),profile("sampling","b",61),profile("sampling","a",80),profile("prep","p",61)];
        let deps=[ProfileDependency{profile:master[0].identity(),requires:vec![req("prep",None)]},
                  ProfileDependency{profile:master[6].identity(),requires:vec![req("router",Some(&["e64"]))]}];
        let mut d=descriptor();
        let plan=resolve_model_scope(&d,&master,&deps).unwrap();
        assert_eq!(plan.selection.profiles.len(),5);
        assert!(plan.materialization_selection().is_ok());
        d.coverage=Coverage::Partial;
        assert!(resolve_model_scope(&d,&master,&deps).unwrap().materialization_selection().is_err());
        d.components[0].requirements=vec![req("router",Some(&["missing"]))];
        assert!(resolve_model_scope(&d,&master,&deps).is_err());
        d.components[0].requirements=vec![req("router",None)];
        assert_eq!(resolve_model_scope(&d,&master,&deps).unwrap().selection.profiles.len(),6);
        d.coverage=Coverage::DeclaredComplete;
        d.unresolved.push("unknown_cache".into());
        assert!(resolve_model_scope(&d,&master,&deps).is_err());
    }
    #[test]
    fn dkr_model_scope_canonical_receipt_and_tamper() -> Result<()> {
        let dir=tempfile::tempdir()?;
        let ptx=b".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry fixture(.param .u32 n) {ret;}\n";
        let manifest=json!({
            "schema":"DKR-KERNEL-MODULE-V1","abi":"dkr.typed.v1",
            "operation":"test.op","implementation_id":"impl.fixture","entry":"fixture",
            "artifact":"fixture.ptx","sha256":digest(ptx),"target_cc":61,"min_cc":61,"max_cc":61,
            "block":[256,1,1],"shared_mem_bytes":0,
            "arguments":[{"kind":"scalar","name":"n","dtype":"u32"}]
        });
        std::fs::write(dir.path().join("fixture.ptx"),ptx)?;
        std::fs::write(dir.path().join("manifest.json"),serde_json::to_vec(&manifest)?)?;
        let catalog=KernelCatalog::discover(dir.path())?;
        let receipt=ScopeReceipt{schema:SCOPE_SCHEMA.into(),device_cc:61,
            scope_id:"084d8d0b35304955c8f6dc69461ef12beee7266500b06bbe97a1561d94747bcc".into(),profile_count:1};
        receipt.verify(&catalog)?;
        std::fs::write(dir.path().join(SCOPE_FILENAME),serde_json::to_vec(&receipt)?)?;
        assert_eq!(read_scope_receipt(dir.path())?,Some(receipt.clone()));
        let mut bad=receipt.clone();bad.device_cc=80;
        assert!(bad.verify(&catalog).is_err());
        let mut bad=receipt.clone();bad.profile_count=2;
        assert!(bad.verify(&catalog).is_err());
        let mut changed=manifest;
        changed["block"]=json!([128,1,1]);
        std::fs::write(dir.path().join("manifest.json"),serde_json::to_vec(&changed)?)?;
        let changed_catalog=KernelCatalog::discover(dir.path())?;
        assert!(receipt.verify(&changed_catalog).is_err());
        Ok(())
    }
    #[test]
    fn dkr_model_scope_latch_rejects_live_switch(){
        let receipt=ScopeReceipt{schema:SCOPE_SCHEMA.into(),device_cc:61,scope_id:"0".repeat(64),profile_count:1};
        let latch=ScopeLatch::default();
        latch.admit(Some(&receipt)).unwrap();
        latch.admit(Some(&receipt)).unwrap();
        assert!(latch.admit(None).is_err());
        let mut other=receipt.clone(); other.scope_id="1".repeat(64);
        assert!(latch.admit(Some(&other)).is_err());
        let unscoped=ScopeLatch::default();
        unscoped.admit(None).unwrap();
        unscoped.admit(None).unwrap();
        assert!(unscoped.admit(Some(&receipt)).is_err());
    }
}
