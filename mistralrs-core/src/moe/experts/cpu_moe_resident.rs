use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender},
        Arc, Mutex, RwLock,
    },
};

use candle_core::{quantized::GgmlDType, DType, Device, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, routed_moe::{RoutedActivation, RoutedExperts, RoutedInput}};

use super::{backends::FastExpertsWeights, forward::{MoEForward, MoEForwardConfig}};

const SLOTS_FLAG: &str = "MISTRALRS_CPU_MOE_RESIDENT_SLOTS";

#[derive(Clone)]
struct ResidentExpert {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

#[derive(Default, Debug, Clone, Copy)]
struct CacheCounters {
    hits: u64,
    misses: u64,
    promotions: u64,
    evictions: u64,
    failed_promotions: u64,
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<u32, Arc<ResidentExpert>>,
    lru: VecDeque<u32>,
    pending: std::collections::HashSet<u32>,
    stats: CacheCounters,
}

impl CacheState {
    fn touch(&mut self, id: u32) {
        self.lru.retain(|cached| *cached != id);
        self.lru.push_back(id);
    }

    fn make_room(&mut self, id: u32, capacity: usize) {
        while self.entries.len() >= capacity && !self.entries.contains_key(&id) {
            if let Some(evicted) = self.lru.pop_front() {
                self.entries.remove(&evicted);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
    }

    fn insert(&mut self, id: u32, expert: Arc<ResidentExpert>) {
        self.entries.insert(id, expert);
        self.touch(id);
        self.stats.promotions += 1;
    }
}

struct CpuJob {
    xs: Tensor,
    indices: Tensor,
    routing: Tensor,
    output_dtype: DType,
    reply: mpsc::Sender<Result<Tensor>>,
}

fn spawn_cpu_worker(
    weights: FastExpertsWeights,
    config: MoEForwardConfig,
) -> Result<SyncSender<CpuJob>> {
    let (cpu_jobs, cpu_rx) = mpsc::sync_channel::<CpuJob>(1);
    std::thread::Builder::new()
        .name("cpu-moe-worker".into())
        .spawn(move || {
            let experts = RoutedExperts {
                gate: weights.fused_gate_proj.as_ref(),
                up: weights.fused_up_proj.as_ref(),
                down: weights.fused_down_proj.as_ref(),
                num_experts: config.num_experts,
            };
            for job in cpu_rx {
                let response = experts.forward(
                    RoutedInput {
                        xs: &job.xs,
                        indices: &job.indices,
                        routing: &job.routing,
                        output_dtype: job.output_dtype,
                        lora: None,
                    },
                    RoutedActivation {
                        fused: crate::ops::glu_activation_type(config.act),
                        fallback: &config.act,
                    },
                );
                let _ = job.reply.send(response);
            }
        })
        .map_err(candle_core::Error::wrap)?;
    Ok(cpu_jobs)
}

#[derive(Default)]
struct HotGroup {
    expert: Option<Arc<ResidentExpert>>,
    tokens: Vec<u32>,
    weights: Vec<f32>,
}

pub(super) struct CpuMoeResident {
    cpu_jobs: SyncSender<CpuJob>,
    prefetch_jobs: SyncSender<u32>,
    cache: Arc<Mutex<CacheState>>,
    execution: Arc<RwLock<()>>,
    device: Device,
    act: crate::layers::Activation,
    num_experts: usize,
    capacity: usize,
    forwards: AtomicU64,
}

impl CpuMoeResident {
    pub(super) fn configured_slots() -> Result<usize> {
        let Ok(value) = std::env::var(SLOTS_FLAG) else {
            return Ok(0);
        };
        Self::parse_slots(&value)
    }

    fn parse_slots(value: &str) -> Result<usize> {
        let slots = value.trim().parse::<usize>().map_err(|_| {
            candle_core::Error::msg(format!("{SLOTS_FLAG} must be a non-negative integer"))
        })?;
        if slots > 32 {
            candle_core::bail!("{SLOTS_FLAG} must not exceed 32 slots per layer");
        }
        Ok(slots)
    }

    pub(super) fn try_new(
        weights: &FastExpertsWeights,
        config: MoEForwardConfig,
        device: &Device,
        capacity: usize,
    ) -> Result<Option<Self>> {
        if capacity == 0 || !device.is_cuda() {
            return Ok(None);
        }
        let all_gguf = [&weights.fused_gate_proj, &weights.fused_up_proj, &weights.fused_down_proj]
            .into_iter()
            .all(|projection| {
                projection.get_qtensor().is_some_and(|q| {
                    matches!(q.shape().dims(), [experts, _, _] if *experts == config.num_experts)
                        && q.dtype() == GgmlDType::Q4K
                        && q.device().is_cpu()
                })
            });
        if !all_gguf {
            tracing::warn!("CPU-MoE resident cache currently requires CPU-packed GGUF Q4K stacks; retaining S0");
            return Ok(None);
        }

        let cache = Arc::new(Mutex::new(CacheState::default()));
        let execution = Arc::new(RwLock::new(()));
        let prefetch_cache = Arc::clone(&cache);
        let prefetch_execution = Arc::clone(&execution);
        let stage_weights = weights.clone();
        let stage_device = device.clone();
        let (prefetch_jobs, prefetch_rx) = mpsc::sync_channel::<u32>(capacity);
        std::thread::Builder::new()
            .name("cpu-moe-prefetch".into())
            .spawn(move || {
                while let Ok(id) = prefetch_rx.recv() {
                    let Ok(_exclusive) = prefetch_execution.write() else {
                        break;
                    };
                    let Ok(mut state) = prefetch_cache.lock() else {
                        break;
                    };
                    state.make_room(id, capacity);
                    drop(state);
                    let stage = || -> Result<Arc<ResidentExpert>> {
                        let load = |layer: &Arc<dyn QuantMethod>| -> Result<Arc<dyn QuantMethod>> {
                            layer.expert_on_device(id as usize, &stage_device)?.ok_or_else(|| {
                                candle_core::Error::msg("GGUF expert extraction unavailable")
                            })
                        };
                        Ok(Arc::new(ResidentExpert {
                            gate: load(&stage_weights.fused_gate_proj)?,
                            up: load(&stage_weights.fused_up_proj)?,
                            down: load(&stage_weights.fused_down_proj)?,
                        }))
                    };
                    let result = stage();
                    let Ok(mut state) = prefetch_cache.lock() else {
                        break;
                    };
                    state.pending.remove(&id);
                    match result {
                        Ok(expert) => state.insert(id, expert),
                        Err(err) => {
                            state.stats.failed_promotions += 1;
                            tracing::warn!("CPU-MoE expert {id} prefetch failed: {err}");
                        }
                    }
                }
            })
            .map_err(candle_core::Error::wrap)?;

        let cpu_jobs = spawn_cpu_worker(weights.clone(), config)?;

        tracing::info!(
            "CPU-MoE per-expert cache enabled: {} slots/layer, async CPU worker, GGUF weights",
            capacity
        );
        Ok(Some(Self {
            cpu_jobs,
            prefetch_jobs,
            cache,
            execution,
            device: device.clone(),
            act: config.act,
            num_experts: config.num_experts,
            capacity,
            forwards: AtomicU64::new(0),
        }))
    }

    fn request_prefetch(&self, id: u32) -> Result<()> {
        let mut state = self.cache.lock().map_err(|_| candle_core::Error::msg("CPU-MoE cache lock poisoned"))?;
        if state.entries.contains_key(&id) || !state.pending.insert(id) {
            return Ok(());
        }
        match self.prefetch_jobs.try_send(id) {
            Ok(()) => Ok(()),
            Err(err) => {
                state.pending.remove(&id);
                if matches!(err, mpsc::TrySendError::Disconnected(_)) {
                    candle_core::bail!("CPU-MoE expert prefetch worker disconnected");
                }
                Ok(())
            }
        }
    }

    pub(super) fn forward(&self, forward: &MoEForward) -> Result<Tensor> {
        let (tokens, hidden) = forward.xs_flat.dims2()?;
        let indices = forward.topk_ids.to_device(&Device::Cpu)?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let weights = forward.topk_weights.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let gpu_execution = self.execution.read().map_err(|_| candle_core::Error::msg("CPU-MoE execution lock poisoned"))?;

        let mut hot: BTreeMap<u32, HotGroup> = BTreeMap::new();
        let mut cold = Vec::<(u32, u32, f32)>::new();
        let mut missed = HashMap::<u32, f32>::new();
        {
            let mut state = self.cache.lock().map_err(|_| candle_core::Error::msg("CPU-MoE cache lock poisoned"))?;
            for token in 0..tokens {
                for (&id, &weight) in indices[token].iter().zip(weights[token].iter()) {
                    if id as usize >= self.num_experts {
                        candle_core::bail!("CPU-MoE routed expert {id} out of bounds");
                    }
                    if let Some(expert) = state.entries.get(&id).cloned() {
                        state.stats.hits += 1;
                        state.touch(id);
                        let group = hot.entry(id).or_default();
                        group.expert = Some(expert);
                        group.tokens.push(token as u32);
                        group.weights.push(weight);
                    } else {
                        state.stats.misses += 1;
                        cold.push((token as u32, id, weight));
                        missed.entry(id).and_modify(|best| *best = (*best).max(weight)).or_insert(weight);
                    }
                }
            }
        }

        let mut cpu_reply = None;
        if !cold.is_empty() {
            let cpu_xs = forward.xs_flat.to_device(&Device::Cpu)?;
            let token_ids = cold.iter().map(|(token, _, _)| *token).collect::<Vec<_>>();
            let token_tensor = Tensor::from_vec(token_ids, cold.len(), &Device::Cpu)?;
            let selected = cpu_xs.index_select(&token_tensor, 0)?;
            let cpu_ids = Tensor::from_vec(cold.iter().map(|(_, id, _)| *id).collect::<Vec<_>>(), (cold.len(), 1), &Device::Cpu)?;
            let cpu_weights = Tensor::from_vec(cold.iter().map(|(_, _, weight)| *weight).collect::<Vec<_>>(), (cold.len(), 1), &Device::Cpu)?
                .to_dtype(forward.topk_weights.dtype())?;
            let (sender, receiver) = mpsc::channel();
            self.cpu_jobs.send(CpuJob {
                xs: selected,
                indices: cpu_ids,
                routing: cpu_weights,
                output_dtype: DType::F32,
                reply: sender,
            }).map_err(candle_core::Error::wrap)?;
            cpu_reply = Some(receiver);
        }
        let mut candidates = missed.into_iter().collect::<Vec<_>>();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (id, _) in candidates.into_iter().take(self.capacity) {
            self.request_prefetch(id)?;
        }

        let mut output = Tensor::zeros((tokens, hidden), DType::F32, &self.device)?;
        for group in hot.values() {
            let expert = group.expert.as_ref().expect("hot group has expert weights");
            let token_ids = Tensor::from_vec(group.tokens.clone(), group.tokens.len(), &self.device)?;
            let xs = forward.xs_flat.index_select(&token_ids, 0)?;
            let gate = expert.gate.forward(&xs)?;
            let up = expert.up.forward(&xs)?;
            let activated = if let Some(fused) = crate::ops::glu_activation_type(self.act) {
                mistralrs_quant::fused_glu(&gate, &up, fused)?
            } else {
                (&up * gate.apply(&self.act)?)?
            };
            let ys = expert.down.forward(&activated)?.to_dtype(DType::F32)?;
            let route_weights = Tensor::from_vec(group.weights.clone(), group.weights.len(), &self.device)?.unsqueeze(1)?;
            let weighted = ys.broadcast_mul(&route_weights)?;
            output = output.index_add(&token_ids, &weighted, 0)?;
        }
        drop(hot);
        drop(gpu_execution);

        if let Some(receiver) = cpu_reply {
            let cpu_result = receiver.recv().map_err(candle_core::Error::wrap)??;
            let token_ids = Tensor::from_vec(cold.iter().map(|(token, _, _)| *token).collect::<Vec<_>>(), cold.len(), &self.device)?;
            let ys = cpu_result.to_device(&self.device)?.to_dtype(DType::F32)?;
            output = output.index_add(&token_ids, &ys, 0)?;
        }

        let calls = self.forwards.fetch_add(1, Ordering::Relaxed) + 1;
        if calls.is_multiple_of(128) {
            if let Ok(state) = self.cache.lock() {
                tracing::info!(
                    "CPU-MoE cache: calls={} hits={} misses={} promotions={} evictions={} failures={} resident={}",
                    calls, state.stats.hits, state.stats.misses, state.stats.promotions,
                    state.stats.evictions, state.stats.failed_promotions, state.entries.len()
                );
            }
        }
        output.to_dtype(forward.original_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Linear;
    use mistralrs_quant::{QuantMethodConfig, UnquantLinear};

    #[test]
    fn lru_touches_are_stable_and_unique() {
        let mut state = CacheState::default();
        state.touch(4);
        state.touch(1);
        state.touch(4);
        assert_eq!(state.lru, VecDeque::from([1, 4]));
    }

    #[test]
    fn slot_limit_is_explicit_opt_in() {
        assert_eq!(CpuMoeResident::parse_slots("0").unwrap(), 0);
        assert_eq!(CpuMoeResident::parse_slots(" 2 ").unwrap(), 2);
        assert!(CpuMoeResident::parse_slots("33").is_err());
        assert!(CpuMoeResident::parse_slots("-1").is_err());
        assert!(CpuMoeResident::parse_slots("garbage").is_err());
    }

    #[test]
    fn async_cpu_worker_preserves_individual_routed_contributions() -> Result<()> {
        let device = &Device::Cpu;
        let linear = |value: f32| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(Tensor::full(value, (2, 8, 8), device)?, None),
            ))?))
        };
        let weights = FastExpertsWeights {
            fused_gate_proj: linear(0.25)?,
            fused_up_proj: linear(0.5)?,
            fused_down_proj: linear(0.125)?,
            sharded: false,
        };
        let config = MoEForwardConfig {
            num_experts: 2,
            num_experts_per_tok: 2,
            act: crate::layers::Activation::Silu,
        };
        let sender = spawn_cpu_worker(weights.clone(), config)?;
        let xs = Tensor::ones((2, 8), DType::F32, device)?;
        let ids = Tensor::new(&[[0u32], [1u32]], device)?;
        let routing = Tensor::new(&[[0.75f32], [0.25f32]], device)?;
        let (reply, result) = mpsc::channel();
        sender.send(CpuJob {
            xs: xs.clone(),
            indices: ids.clone(),
            routing: routing.clone(),
            output_dtype: DType::F32,
            reply,
        }).map_err(candle_core::Error::wrap)?;
        let actual = result.recv().map_err(candle_core::Error::wrap)??;
        let reference = RoutedExperts {
            gate: weights.fused_gate_proj.as_ref(),
            up: weights.fused_up_proj.as_ref(),
            down: weights.fused_down_proj.as_ref(),
            num_experts: 2,
        }.gather(RoutedInput {
            xs: &xs,
            indices: &ids,
            routing: &routing,
            output_dtype: DType::F32,
            lora: None,
        }, RoutedActivation {
            fused: crate::ops::glu_activation_type(config.act),
            fallback: &config.act,
        })?;
        assert_eq!(actual.to_vec2::<f32>()?, reference.to_vec2::<f32>()?);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cached_and_cpu_missed_routes_match_q4k_baseline() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QTensor};
        use mistralrs_quant::GgufMatMul;
        use std::time::Duration;

        const HIDDEN: usize = 256;
        let gpu = Device::new_cuda(0)?;
        let cpu = Device::Cpu;
        let projection = |phase: usize| -> Result<Arc<dyn QuantMethod>> {
            let values = (0..2 * HIDDEN * HIDDEN)
                .map(|i| ((i + phase) % 79) as f32 / 173.0 - 0.23)
                .collect::<Vec<_>>();
            let stack = Tensor::from_vec(values, (2, HIDDEN, HIDDEN), &cpu)?;
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(QTensor::quantize(&stack, GgmlDType::Q4K)?),
                b: None,
            })?))
        };
        let weights = FastExpertsWeights {
            fused_gate_proj: projection(2)?,
            fused_up_proj: projection(7)?,
            fused_down_proj: projection(13)?,
            sharded: false,
        };
        let config = MoEForwardConfig {
            num_experts: 2,
            num_experts_per_tok: 2,
            act: crate::layers::Activation::Silu,
        };
        let resident = CpuMoeResident::try_new(&weights, config, &gpu, 1)?
            .expect("CUDA Q4K can be staged per expert");
        let xs_cpu = Tensor::from_vec(
            (0..HIDDEN).map(|i| i as f32 / 121.0 - 1.0).collect(),
            (1, HIDDEN),
            &cpu,
        )?;
        let ids_cpu = Tensor::new(&[[0u32, 1]], &cpu)?;
        let routing_cpu = Tensor::new(&[[0.75f32, 0.25]], &cpu)?;
        let xs_gpu = xs_cpu.to_device(&gpu)?;
        let ids_gpu = ids_cpu.to_device(&gpu)?;
        let routing_gpu = routing_cpu.to_device(&gpu)?;
        let forward = MoEForward {
            xs_flat: &xs_gpu,
            topk_weights: &routing_gpu,
            topk_ids: &ids_gpu,
            original_dtype: DType::F32,
            shape: super::super::forward::MoEForwardShape::new(1, 1, HIDDEN),
            lora: None,
        };
        let expected = RoutedExperts {
            gate: weights.fused_gate_proj.as_ref(),
            up: weights.fused_up_proj.as_ref(),
            down: weights.fused_down_proj.as_ref(),
            num_experts: 2,
        }.forward(RoutedInput {
            xs: &xs_cpu,
            indices: &ids_cpu,
            routing: &routing_cpu,
            output_dtype: DType::F32,
            lora: None,
        }, RoutedActivation {
            fused: crate::ops::glu_activation_type(config.act),
            fallback: &config.act,
        })?;

        resident.forward(&forward)?;
        for _ in 0..200 {
            if resident.cache.lock().map_err(|_| candle_core::Error::msg("cache poisoned"))?
                .stats.promotions > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let actual = resident.forward(&forward)?.to_device(&cpu)?;
        let max_error = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_error < 0.05, "hybrid Q4K numerical deviation: {max_error}");
        let stats = resident.cache.lock().map_err(|_| candle_core::Error::msg("cache poisoned"))?.stats;
        assert!(stats.promotions > 0 && stats.hits > 0 && stats.misses > 0, "{stats:?}");
        Ok(())
    }
}
