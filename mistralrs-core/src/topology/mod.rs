use std::{
    fs,
    io::Read,
    ops::Range,
    path::Path,
    sync::atomic::{AtomicU8, Ordering},
};

use candle_core::Device;
use indexmap::IndexMap;
use itertools::Itertools;
use mistralrs_quant::{GgufCpuRuntimeOptions, IsqType};
use regex::Regex;
use serde::Deserialize;

use crate::parse_isq_value;

const DEVICE_PATTERN: &str = r"^(cpu|cuda\[(\d+)\]|metal\[(\d+)\])$";
const TOPOLOGY_BOOL_UNSET: u8 = 0;
const TOPOLOGY_BOOL_FALSE: u8 = 1;
const TOPOLOGY_BOOL_TRUE: u8 = 2;

static QWEN35_CPU_MOE: AtomicU8 = AtomicU8::new(TOPOLOGY_BOOL_UNSET);
static QWEN35_PROFILE: AtomicU8 = AtomicU8::new(TOPOLOGY_BOOL_UNSET);

#[derive(Deserialize)]
pub struct DeserLayerTopology {
    isq: Option<String>,
    device: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct TopologyRuntime {
    #[serde(
        alias = "MISTRALRS_QWEN35_CPU_MOE",
        alias = "MISTRALRS_CPU_MOE",
        alias = "qwen35-cpu-moe",
        alias = "cpu-moe"
    )]
    pub qwen35_cpu_moe: Option<bool>,
    #[serde(
        alias = "MISTRALRS_QWEN35_PROFILE",
        alias = "MISTRALRS_CPU_PROFILE",
        alias = "qwen35-profile",
        alias = "cpu-profile"
    )]
    pub qwen35_profile: Option<bool>,
    #[serde(
        alias = "MISTRALRS_GGUF_CPU_MOE_EXPERT_CACHE",
        alias = "gguf-cpu-moe-expert-cache"
    )]
    pub gguf_cpu_moe_expert_cache: Option<usize>,
    #[serde(
        alias = "MISTRALRS_GGUF_CPU_MOE_Q4_1_EXPERT_CACHE",
        alias = "gguf-cpu-moe-q4-1-expert-cache"
    )]
    pub gguf_cpu_moe_q4_1_expert_cache: Option<usize>,
    #[serde(
        alias = "MISTRALRS_GGUF_CPU_MOE_Q4K_EXPERT_CACHE",
        alias = "gguf-cpu-moe-q4k-expert-cache"
    )]
    pub gguf_cpu_moe_q4k_expert_cache: Option<usize>,
    #[serde(
        alias = "MISTRALRS_GGUF_CPU_MOE_PARALLEL_TOPK",
        alias = "gguf-cpu-moe-parallel-topk"
    )]
    pub gguf_cpu_moe_parallel_topk: Option<bool>,
    #[serde(alias = "MISTRALRS_GGUF_CPU_Q4K_MATMUL", alias = "gguf-cpu-q4k-matmul")]
    pub gguf_cpu_q4k_matmul: Option<bool>,
    #[serde(
        alias = "MISTRALRS_GGUF_CPU_Q4K_MATMUL_CACHE",
        alias = "gguf-cpu-q4k-matmul-cache"
    )]
    pub gguf_cpu_q4k_matmul_cache: Option<usize>,
    #[serde(
        alias = "MISTRALRS_GGUF_CPU_Q4K_MATMUL_MAX_ROWS",
        alias = "gguf-cpu-q4k-matmul-max-rows"
    )]
    pub gguf_cpu_q4k_matmul_max_rows: Option<usize>,
}

#[derive(Default, Deserialize)]
pub struct DeserTopology {
    #[serde(default)]
    runtime: TopologyRuntime,
    #[serde(default)]
    layers: IndexMap<String, DeserLayerTopology>,
    #[serde(flatten)]
    flat_layers: IndexMap<String, DeserLayerTopology>,
}

#[derive(Clone, Debug)]
pub struct LayerTopology {
    pub isq: Option<IsqType>,
    pub device: Option<Device>,
}

#[derive(PartialEq, Eq, Debug)]
struct CustomRange {
    start: usize,
    end: usize,
    index: usize,
}

impl From<CustomRange> for Range<usize> {
    fn from(value: CustomRange) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

impl Ord for CustomRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Order based on end position followed by declaration order so later ranges override
        self.end
            .cmp(&other.end)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for CustomRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub layers: Vec<Option<LayerTopology>>,
    pub patterns: Vec<(Regex, LayerTopology)>,
    pub runtime: TopologyRuntime,
}

impl Topology {
    /// Create an empty topology.
    pub fn empty() -> Self {
        Topology {
            layers: Vec::new(),
            patterns: Vec::new(),
            runtime: TopologyRuntime::default(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Topology {
            layers: vec![None; cap],
            patterns: Vec::new(),
            runtime: TopologyRuntime::default(),
        }
    }

    pub fn is_dummy_device_map(&self) -> bool {
        self.layers
            .iter()
            .all(|l| l.is_none() || l.as_ref().is_some_and(|l| l.device.is_none()))
            && self
                .patterns
                .iter()
                .all(|(_, topo)| topo.device.as_ref().is_none())
    }

    pub fn with_range(mut self, range: Range<usize>, layer: LayerTopology) -> Self {
        if self.layers.len() < range.end {
            self.layers
                .extend(vec![None; range.end - self.layers.len()]);
        }
        for i in range.start..range.end {
            self.layers[i] = Some(layer.clone());
        }
        self
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(topology: &str) -> anyhow::Result<Self> {
        let deser: DeserTopology = serde_saphyr::from_str(topology)?;
        let device_regex = Regex::new(DEVICE_PATTERN)?;
        let DeserTopology {
            runtime,
            layers,
            flat_layers,
        } = deser;

        let mut range_layers = Vec::new();
        let mut pattern_layers = Vec::new();
        let mut entries = flat_layers.into_iter().collect::<Vec<_>>();
        entries.extend(layers);
        for (index, (selector, DeserLayerTopology { isq, device })) in
            entries.into_iter().enumerate()
        {
            let parsed_isq = if let Some(isq) = isq {
                Some(parse_isq_value(&isq, None).map_err(anyhow::Error::msg)?)
            } else {
                None
            };

            let parsed_device = if let Some(device) = device {
                let Some(captures) = device_regex.captures(&device) else {
                    anyhow::bail!(
                        "Device specifier must match regex {DEVICE_PATTERN}. Examples: `cpu`, `cuda[ORD]`, `metal[ORD]`"
                    );
                };
                let device = if let Some(val) = captures.get(2).or(captures.get(3)) {
                    let ord = val.as_str().parse::<usize>()?;
                    let device = device.split('[').collect::<Vec<_>>()[0];
                    match device {
                        "cuda" => Device::new_cuda(ord)?,
                        "metal" => Device::new_metal(ord)?,
                        _ => unreachable!(),
                    }
                } else {
                    Device::Cpu
                };

                Some(device)
            } else {
                None
            };

            if selector.starts_with('/') && selector.ends_with('/') && selector.len() >= 2 {
                let pattern = &selector[1..selector.len() - 1];
                let regex = Regex::new(pattern)
                    .map_err(|err| anyhow::anyhow!("Invalid topology regex `{pattern}`: {err}"))?;
                pattern_layers.push((
                    regex,
                    LayerTopology {
                        isq: parsed_isq,
                        device: parsed_device,
                    },
                ));
                continue;
            }

            let (start, end) = if selector.contains('-') {
                // Range (inclusive, exclusive)
                let Some((start, end)) = selector.splitn(2, '-').collect_tuple() else {
                    anyhow::bail!("Topology range segment must follow the format START-END")
                };
                (start.parse::<usize>()?, end.parse::<usize>()?)
            } else {
                // Single layer here
                let layer = selector.parse::<usize>()?;
                (layer, layer + 1)
            };

            if end <= start {
                anyhow::bail!("Topology range end must be > start, got {end} <= {start}");
            }
            let range = CustomRange { start, end, index };

            range_layers.push((
                range,
                LayerTopology {
                    isq: parsed_isq,
                    device: parsed_device,
                },
            ));
        }
        // Sort so that we increase in end points
        range_layers.sort_by(|(r1, _), (r2, _)| r1.cmp(r2));

        let capacity = range_layers.iter().map(|(r, _)| r.end).max().unwrap_or(0);
        let mut this = if capacity == 0 {
            Self::empty()
        } else {
            Self::with_capacity(capacity)
        };
        for (range, layer) in range_layers {
            for i in range.start..range.end {
                this.layers[i] = Some(layer.clone());
            }
        }
        this.patterns = pattern_layers;
        this.runtime = runtime;
        Ok(this)
    }

    pub fn from_reader<R: Read>(mut reader: R) -> anyhow::Result<Self> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        let topology = Self::from_str(&buf)?;
        topology.apply_runtime_options();
        Ok(topology)
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let buf = fs::read_to_string(path)?;
        let topology = Self::from_str(&buf)?;
        topology.apply_runtime_options();
        Ok(topology)
    }

    pub fn from_option_path<P: AsRef<Path>>(path: Option<P>) -> anyhow::Result<Option<Self>> {
        if let Some(path) = path {
            let buf = fs::read_to_string(path)?;
            let topology = Self::from_str(&buf)?;
            topology.apply_runtime_options();
            Ok(Some(topology))
        } else {
            Ok(None)
        }
    }

    pub fn apply_runtime_options(&self) {
        self.runtime.apply();
    }

    pub fn layer_for(&self, layer: usize) -> Option<&LayerTopology> {
        self.layers.get(layer).and_then(|lt| lt.as_ref())
    }

    pub fn match_for_name(&self, name: &str) -> Option<LayerTopology> {
        for (regex, layer) in self.patterns.iter().rev() {
            if regex.is_match(name) {
                return Some(layer.clone());
            }
        }
        None
    }

    pub fn pattern_overrides(&self) -> Vec<(Regex, LayerTopology)> {
        self.patterns
            .iter()
            .rev()
            .map(|(regex, topo)| (regex.clone(), topo.clone()))
            .collect()
    }

    pub fn requires_post_quantization(&self) -> bool {
        self.layers.iter().any(|layer| {
            layer
                .as_ref()
                .is_some_and(|layer| layer.isq.is_some() || layer.device.is_some())
        })
    }
}

impl TopologyRuntime {
    fn has_any(&self) -> bool {
        self.qwen35_cpu_moe.is_some()
            || self.qwen35_profile.is_some()
            || self.gguf_cpu_moe_expert_cache.is_some()
            || self.gguf_cpu_moe_q4_1_expert_cache.is_some()
            || self.gguf_cpu_moe_q4k_expert_cache.is_some()
            || self.gguf_cpu_moe_parallel_topk.is_some()
            || self.gguf_cpu_q4k_matmul.is_some()
            || self.gguf_cpu_q4k_matmul_cache.is_some()
            || self.gguf_cpu_q4k_matmul_max_rows.is_some()
    }

    fn apply(&self) {
        if !self.has_any() {
            return;
        }

        store_optional_bool(&QWEN35_CPU_MOE, self.qwen35_cpu_moe);
        store_optional_bool(&QWEN35_PROFILE, self.qwen35_profile);
        mistralrs_quant::set_gguf_cpu_runtime_options(GgufCpuRuntimeOptions {
            cpu_moe_expert_cache: self.gguf_cpu_moe_expert_cache,
            cpu_moe_q4_1_expert_cache: self.gguf_cpu_moe_q4_1_expert_cache,
            cpu_moe_q4k_expert_cache: self.gguf_cpu_moe_q4k_expert_cache,
            cpu_moe_parallel_topk: self.gguf_cpu_moe_parallel_topk,
            cpu_q4k_matmul: self.gguf_cpu_q4k_matmul,
            cpu_q4k_matmul_cache: self.gguf_cpu_q4k_matmul_cache,
            cpu_q4k_matmul_max_rows: self.gguf_cpu_q4k_matmul_max_rows,
        });
    }
}

fn store_optional_bool(atom: &AtomicU8, value: Option<bool>) {
    atom.store(
        match value {
            None => TOPOLOGY_BOOL_UNSET,
            Some(false) => TOPOLOGY_BOOL_FALSE,
            Some(true) => TOPOLOGY_BOOL_TRUE,
        },
        Ordering::Relaxed,
    );
}

fn load_optional_bool(atom: &AtomicU8) -> Option<bool> {
    match atom.load(Ordering::Relaxed) {
        TOPOLOGY_BOOL_FALSE => Some(false),
        TOPOLOGY_BOOL_TRUE => Some(true),
        _ => None,
    }
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|value| !value.is_empty() && value != "0")
}

pub(crate) fn qwen35_cpu_moe_enabled() -> bool {
    load_optional_bool(&QWEN35_CPU_MOE)
        .or_else(|| env_bool("MISTRALRS_CPU_MOE"))
        .or_else(|| env_bool("MISTRALRS_QWEN35_CPU_MOE"))
        .unwrap_or(false)
}

pub(crate) fn qwen35_profile_enabled() -> bool {
    load_optional_bool(&QWEN35_PROFILE)
        .or_else(|| env_bool("MISTRALRS_CPU_PROFILE"))
        .or_else(|| env_bool("MISTRALRS_QWEN35_PROFILE"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_isq(topology: &Topology, layer: usize) -> Option<IsqType> {
        topology
            .layer_for(layer)
            .and_then(|lt| lt.isq.as_ref().copied())
    }

    #[test]
    fn highest_end_range_overrides_lower_end() {
        let yaml = "0-4:\n  isq: Q4K\n2-6:\n  isq: Q6K\n";
        let topology = Topology::from_str(yaml).expect("topology parses");

        assert_eq!(layer_isq(&topology, 0), Some(IsqType::Q4K));
        assert_eq!(layer_isq(&topology, 2), Some(IsqType::Q6K));
        assert_eq!(layer_isq(&topology, 5), Some(IsqType::Q6K));
    }

    #[test]
    fn later_range_with_same_end_wins() {
        let yaml = "0-4:\n  isq: Q4K\n2-4:\n  isq: Q3K\n";
        let topology = Topology::from_str(yaml).expect("topology parses");

        assert_eq!(layer_isq(&topology, 1), Some(IsqType::Q4K));
        assert_eq!(layer_isq(&topology, 2), Some(IsqType::Q3K));
        assert_eq!(layer_isq(&topology, 3), Some(IsqType::Q3K));
    }

    #[test]
    fn regex_overrides_respect_declaration_order() {
        let yaml = r#"'/ffn\./':
  isq: Q4K
'/ffn\.weight$/':
  isq: Q6K
"#;
        let topology = Topology::from_str(yaml).expect("topology parses");

        let match_exact = topology
            .match_for_name("model.layers.2.ffn.weight")
            .expect("regex match");
        assert_eq!(match_exact.isq, Some(IsqType::Q6K));

        let overrides = topology.pattern_overrides();
        assert_eq!(overrides.len(), 2);
        assert_eq!(overrides[0].0.as_str(), "ffn\\.weight$");
        assert_eq!(overrides[1].0.as_str(), "ffn\\.");
    }

    #[test]
    fn runtime_and_nested_layers_parse() {
        let yaml = r#"
runtime:
  cpu_moe: true
  cpu_profile: true
  gguf_cpu_moe_q4k_expert_cache: 2048
layers:
  0-2:
    isq: Q4K
    device: cpu
"#;
        let topology = Topology::from_str(yaml).expect("topology parses");

        assert_eq!(topology.runtime.qwen35_cpu_moe, Some(true));
        assert_eq!(topology.runtime.qwen35_profile, Some(true));
        assert_eq!(topology.runtime.gguf_cpu_moe_q4k_expert_cache, Some(2048));
        assert_eq!(layer_isq(&topology, 0), Some(IsqType::Q4K));
        assert!(matches!(
            topology.layer_for(0).and_then(|lt| lt.device.as_ref()),
            Some(&Device::Cpu)
        ));
    }

    #[test]
    fn match_for_name_returns_none_when_unmatched() {
        let yaml = "0-2:\n  isq: Q4K\n";
        let topology = Topology::from_str(yaml).expect("topology parses");
        assert!(topology.match_for_name("transformer.wte.weight").is_none());
    }
}
