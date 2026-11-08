use crate::{
    cuda::{compute_capability, PreparedLinear},
    profile::{BoundDecision, LoadedProfile, PROFILE_ENV},
};
use anyhow::{ensure, Context, Result};
use candle_core::{cuda_backend::DeviceId, DType, Tensor};
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

static SESSION: OnceLock<Result<Option<Arc<Session>>, String>> = OnceLock::new();

struct Session {
    profile: LoadedProfile,
    modules: Mutex<HashMap<(DeviceId, String), Arc<PreparedLinear>>>,
    prepared_layers: AtomicU64,
    devices: Mutex<HashMap<DeviceId, (u32, String)>>,
}

type Selected = (Arc<BoundDecision>, Arc<PreparedLinear>);

pub struct LayerPlan {
    decisions: HashMap<usize, Selected>,
    k: usize,
    n: usize,
}

impl std::fmt::Debug for LayerPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerPlan")
            .field("n", &self.n)
            .field("k", &self.k)
            .field("row_counts", &self.decisions.keys())
            .finish()
    }
}

fn configured() -> Result<Option<&'static Arc<Session>>> {
    let result = SESSION.get_or_init(|| {
        let Some(path) = std::env::var_os(PROFILE_ENV) else {
            return Ok(None);
        };
        LoadedProfile::read(Path::new(&path))
            .map(|profile| {
                Some(Arc::new(Session {
                    profile,
                    modules: Mutex::new(HashMap::new()),
                    prepared_layers: AtomicU64::new(0),
                    devices: Mutex::new(HashMap::new()),
                }))
            })
            .map_err(|e| format!("{PROFILE_ENV}: {e:#}"))
    });
    match result {
        Ok(session) => Ok(session.as_ref()),
        Err(error) => anyhow::bail!("{error}"),
    }
}

impl LayerPlan {
    pub fn for_weight(weight: &Tensor) -> Result<Option<Self>> {
        if weight.dtype() != DType::F32 || !weight.device().is_cuda() || weight.rank() != 2 {
            return Ok(None);
        }
        let Some(session) = configured()? else {
            return Ok(None);
        };
        let (n, k) = weight.dims2()?;
        let device = weight.device().as_cuda_device()?;
        let cc = compute_capability(weight.device())?;
        ensure!(
            cc == session.profile.profile.target_cc,
            "device architecture differs from exact profile target"
        );
        let uuid = device.cuda_stream().context().uuid()?;
        let bytes: Vec<u8> = uuid.bytes.iter().map(|b| *b as u8).collect();
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let uuid = format!(
            "GPU-{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        );
        session
            .devices
            .lock()
            .map_err(|_| anyhow::anyhow!("device cache poisoned"))?
            .entry(device.id())
            .or_insert((cc, uuid));
        let mut modules = session
            .modules
            .lock()
            .map_err(|_| anyhow::anyhow!("provider module cache poisoned"))?;
        let mut decisions = HashMap::new();
        for bound in &session.profile.decisions {
            let [m, dn, dk] = bound.decision.shape;
            if (dn, dk) != (n, k) {
                continue;
            }
            let key = (device.id(), bound.decision.manifest_sha256.clone());
            let prepared = if let Some(prepared) = modules.get(&key) {
                prepared.clone()
            } else {
                let prepared = Arc::new(PreparedLinear::load_ref(&bound.module, weight.device())?);
                modules.insert(key, prepared.clone());
                prepared
            };
            decisions.insert(m, (bound.clone(), prepared));
        }
        ensure!(
            !decisions.is_empty(),
            "required ASD profile has no dense F32 decision for N={n}, K={k}"
        );
        session.prepared_layers.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Self { decisions, k, n }))
    }

    pub fn run(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let last = input
            .rank()
            .checked_sub(1)
            .context("linear input must have at least one dimension")?;
        ensure!(
            input.dtype() == DType::F32 && input.dim(last)? == self.k,
            "profile linear input dtype/K mismatch"
        );
        let m: usize = input.dims()[..last].iter().product();
        let (bound, prepared) = self.decisions.get(&m).with_context(|| {
            format!(
                "required ASD profile has no exact linear decision for M={m}, N={}, K={}",
                self.n, self.k
            )
        })?;
        let flat = input.force_contiguous()?.reshape((m, self.k))?;
        let mut output = prepared.run(&flat, weight)?;
        if let Some(bias) = bias {
            output = output.broadcast_add(bias)?;
        }
        let mut shape = input.dims().to_vec();
        shape[last] = self.n;
        let output = output.reshape(shape)?;
        bound.submissions.fetch_add(1, Ordering::Relaxed);
        Ok(output)
    }
}

pub fn evidence() -> serde_json::Value {
    match SESSION.get() {
        Some(Ok(Some(session))) => {
            let mut evidence = session.profile.evidence();
            evidence["prepared_layers"] = session.prepared_layers.load(Ordering::Relaxed).into();
            evidence["resolved_modules"] = session.modules.lock().unwrap().len().into();
            evidence["fallback_used"] = false.into();
            evidence["devices"] = serde_json::json!(session
                .devices
                .lock()
                .unwrap()
                .values()
                .map(|(cc, uuid)| serde_json::json!({"compute_capability":cc,"gpu_uuid":uuid}))
                .collect::<Vec<_>>());
            evidence
        }
        Some(Err(error)) => serde_json::json!({"configuration_error":error}),
        _ => serde_json::json!({"profile_id":null, "prepared_layers":0, "decisions":[]}),
    }
}
