use std::{
    collections::HashMap,
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::{
    config::{KvCacheLayout, ModelConfigLike},
    kvarn_cache, turboquant_cache,
};

#[derive(Debug)]
pub struct DecodedKVCache {
    pub key_cache: Tensor,
    pub value_cache: Tensor,

    // CPU-side metadata for LRU/remap.
    // physical block id -> decoded cache slot
    pub physical_to_slot: HashMap<u32, usize>,

    // decoded cache slot -> physical block id
    pub slot_to_physical: Vec<Option<u32>>,

    // simple monotonically increasing LRU clock
    pub lru_clock: Vec<u64>,
    pub clock: u64,

    pub block_size: usize,
    pub num_kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub dtype: DType,
}

impl DecodedKVCache {
    pub fn num_slots(&self) -> usize {
        self.slot_to_physical.len()
    }

    pub fn clear_mapping(&mut self) {
        self.physical_to_slot.clear();
        self.slot_to_physical.fill(None);
        self.lru_clock.fill(0);
        self.clock = 0;
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq))]
pub enum PagedCacheType {
    #[serde(alias = "auto")]
    Auto(),
    #[serde(alias = "f8e4m3")]
    F8E4M3(),
    #[serde(alias = "turboquant", alias = "tq")]
    TurboQuant(),
    #[serde(alias = "turboquant_cached")]
    TurboQuantCached { decoded_cache_mb: usize },
    #[serde(alias = "kvarn", alias = "kvarn_k4v2_g128")]
    KVarN(),
}

impl Default for PagedCacheType {
    fn default() -> Self {
        Self::Auto()
    }
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> Result<DType> {
        match self {
            PagedCacheType::F8E4M3() => Ok(DType::F8E4M3),
            PagedCacheType::Auto() => Ok(act_dtype),
            PagedCacheType::TurboQuant() => Ok(DType::U8),
            PagedCacheType::TurboQuantCached { .. } => Ok(DType::U8),
            PagedCacheType::KVarN() => Ok(DType::U8),
        }
    }

    pub fn is_turboquant(&self) -> bool {
        matches!(
            self,
            PagedCacheType::TurboQuant() | PagedCacheType::TurboQuantCached { .. }
        )
    }

    pub fn turboquant_cache_decoded_size(&self) -> Option<usize> {
        match self {
            PagedCacheType::TurboQuantCached { decoded_cache_mb } => Some(*decoded_cache_mb),
            _ => None,
        }
    }

    pub fn is_kvarn(&self) -> bool {
        matches!(self, PagedCacheType::KVarN())
    }

    pub fn bytes_per_block_all_layers(
        &self,
        act_dtype: DType,
        model_config: &dyn ModelConfigLike,
        block_size: usize,
    ) -> Result<usize> {
        match (self, model_config.kv_cache_layout()) {
            (
                PagedCacheType::TurboQuant(),
                KvCacheLayout::Standard
                | KvCacheLayout::StandardNoFlashInfer
                | KvCacheLayout::FlashInferHnd,
            ) => {
                let mut bytes_per_token = 0;
                for layer_idx in 0..model_config.num_layers() {
                    let heads = model_config.num_kv_heads_for_layer(layer_idx);
                    bytes_per_token += heads
                        * (turboquant_cache::row_bytes(
                            model_config.k_head_dim_for_layer(layer_idx),
                        )? + turboquant_cache::row_bytes(
                            model_config.v_head_dim_for_layer(layer_idx),
                        )?);
                }
                Ok(bytes_per_token * block_size)
            }
            (
                PagedCacheType::TurboQuantCached { .. },
                KvCacheLayout::Standard
                | KvCacheLayout::StandardNoFlashInfer
                | KvCacheLayout::FlashInferHnd,
            ) => {
                let mut bytes_per_token = 0;
                for layer_idx in 0..model_config.num_layers() {
                    let heads = model_config.num_kv_heads_for_layer(layer_idx);
                    bytes_per_token += heads
                        * (turboquant_cache::row_bytes(
                            model_config.k_head_dim_for_layer(layer_idx),
                        )? + turboquant_cache::row_bytes(
                            model_config.v_head_dim_for_layer(layer_idx),
                        )?);
                }
                Ok(bytes_per_token * block_size)
            }

            (
                PagedCacheType::TurboQuant() | PagedCacheType::TurboQuantCached { .. },
                KvCacheLayout::Mla { .. },
            ) => {
                candle_core::bail!(
                    "TurboQuantCached paged KV cache does not support MLA cache layout."
                )
            }
            (
                PagedCacheType::KVarN(),
                KvCacheLayout::Standard
                | KvCacheLayout::StandardNoFlashInfer
                | KvCacheLayout::FlashInferHnd,
            ) => {
                let mut bytes_per_block = 0;
                for layer_idx in 0..model_config.num_layers() {
                    let heads = model_config.num_kv_heads_for_layer(layer_idx);
                    bytes_per_block += heads
                        * (kvarn_cache::key_record_bytes(
                            model_config.k_head_dim_for_layer(layer_idx),
                            block_size,
                        )? + kvarn_cache::value_record_bytes(
                            model_config.v_head_dim_for_layer(layer_idx),
                            block_size,
                        )?);
                }
                Ok(bytes_per_block)
            }
            (PagedCacheType::KVarN(), KvCacheLayout::Mla { .. }) => {
                candle_core::bail!("KVarN paged KV cache does not support MLA cache layout.")
            }
            (
                _,
                KvCacheLayout::Standard
                | KvCacheLayout::StandardNoFlashInfer
                | KvCacheLayout::FlashInferHnd,
            ) => {
                let dtype = self.to_dtype(act_dtype)?;
                let mut elements_per_token = 0;
                for layer_idx in 0..model_config.num_layers() {
                    elements_per_token += model_config.num_kv_heads_for_layer(layer_idx)
                        * (model_config.k_head_dim_for_layer(layer_idx)
                            + model_config.v_head_dim_for_layer(layer_idx));
                }
                Ok(elements_per_token * block_size * dtype.size_in_bytes())
            }
            (
                _,
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                },
            ) => {
                let dtype = self.to_dtype(act_dtype)?;
                Ok(model_config.num_layers()
                    * block_size
                    * (kv_lora_rank + kpe_head_dim)
                    * dtype.size_in_bytes())
            }
        }
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto()),
            "f8e4m3" => Ok(Self::F8E4M3()),
            "turboquant" | "tq" => Ok(Self::TurboQuant()),
            "kvarn" | "kvarn_k4v2_g128" => Ok(Self::KVarN()),
            s if s.contains("turboquant_cached:") => {
                // Default to 512MB decoded cache size if not specified, which is a reasonable starting point for many models and fits within the VRAM constraints of most GPUs when using an 8-bit quantized cache
                Ok(Self::TurboQuantCached { decoded_cache_mb: s.strip_prefix("turboquant_cached:").unwrap_or("512").parse().unwrap_or(512) })
            },
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected `auto`, `f8e4m3`, `turboquant`, or `kvarn`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
    pub kv_cache_group_ids: Vec<u32>,
}

pub type KVCache = (Tensor, Tensor);

// pub struct CacheEngine {
//     gpu_cache: Arc<Mutex<Vec<KVCache>>>,
// }

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,

    // Present only for `turboquant_cached:<mb>` / `turboquant_legacy:<mb>`.
    // One decoded cache per layer.
    decoded_gpu_cache: Option<Arc<Mutex<Vec<DecodedKVCache>>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let cache_dtype = cache_config.cache_type.to_dtype(dtype)?;

        let gpu_cache = Self::allocate_gpu_cache(
            model_config,
            cache_config,
            cache_dtype,
            device,
            layer_devices.clone(),
        )?;

        let decoded_gpu_cache = Self::allocate_decoded_gpu_cache(
            model_config,
            cache_config,
            dtype,
            device,
            layer_devices,
        )?;

        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(gpu_cache)),
            decoded_gpu_cache: decoded_gpu_cache.map(|x| Arc::new(Mutex::new(x))),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    pub fn get_decoded_kv_cache(&self) -> Option<MutexGuard<'_, Vec<DecodedKVCache>>> {
        self.decoded_gpu_cache
            .as_ref()
            .map(|cache| cache.lock().expect("decoded KV cache mutex was poisoned"))
    }

    pub fn has_decoded_kv_cache(&self) -> bool {
        self.decoded_gpu_cache.is_some()
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let kv_cache_layout = model_config.kv_cache_layout();
        let mut gpu_cache = Vec::new();

        for (layer_idx, device) in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
            .enumerate()
        {
            let (key_blocks, value_blocks) = match (cache_config.cache_type, kv_cache_layout) {
                (
                    PagedCacheType::TurboQuant() | PagedCacheType::TurboQuantCached { .. },
                    KvCacheLayout::Standard
                    | KvCacheLayout::StandardNoFlashInfer
                    | KvCacheLayout::FlashInferHnd,
                ) => {
                    let num_heads = model_config.num_kv_heads_for_layer(layer_idx);
                    let key_row_bytes =
                        turboquant_cache::row_bytes(model_config.k_head_dim_for_layer(layer_idx))?;
                    let value_row_bytes =
                        turboquant_cache::row_bytes(model_config.v_head_dim_for_layer(layer_idx))?;
                    let key_blocks = Tensor::zeros(
                        (
                            cache_config.num_gpu_blocks,
                            num_heads,
                            cache_config.block_size,
                            key_row_bytes,
                        ),
                        DType::U8,
                        device,
                    )?;
                    let value_blocks = Tensor::zeros(
                        (
                            cache_config.num_gpu_blocks,
                            num_heads,
                            cache_config.block_size,
                            value_row_bytes,
                        ),
                        DType::U8,
                        device,
                    )?;
                    (key_blocks, value_blocks)
                }
                (
                    PagedCacheType::TurboQuant() | PagedCacheType::TurboQuantCached { .. },
                    KvCacheLayout::Mla { .. },
                ) => {
                    candle_core::bail!(
                        "TurboQuant paged KV cache does not support MLA cache layout."
                    )
                }
                (
                    PagedCacheType::KVarN(),
                    KvCacheLayout::Standard
                    | KvCacheLayout::StandardNoFlashInfer
                    | KvCacheLayout::FlashInferHnd,
                ) => {
                    let num_heads = model_config.num_kv_heads_for_layer(layer_idx);
                    let key_record_bytes = kvarn_cache::key_record_bytes(
                        model_config.k_head_dim_for_layer(layer_idx),
                        cache_config.block_size,
                    )?;
                    let value_record_bytes = kvarn_cache::value_record_bytes(
                        model_config.v_head_dim_for_layer(layer_idx),
                        cache_config.block_size,
                    )?;
                    let key_blocks = Tensor::zeros(
                        (cache_config.num_gpu_blocks, num_heads, key_record_bytes),
                        DType::U8,
                        device,
                    )?;
                    let value_blocks = Tensor::zeros(
                        (cache_config.num_gpu_blocks, num_heads, value_record_bytes),
                        DType::U8,
                        device,
                    )?;
                    (key_blocks, value_blocks)
                }
                (PagedCacheType::KVarN(), KvCacheLayout::Mla { .. }) => {
                    candle_core::bail!("KVarN paged KV cache does not support MLA cache layout.")
                }
                (
                    _,
                    KvCacheLayout::Standard
                    | KvCacheLayout::StandardNoFlashInfer
                    | KvCacheLayout::FlashInferHnd,
                ) => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                (
                    _,
                    KvCacheLayout::Mla {
                        kv_lora_rank,
                        kpe_head_dim,
                    },
                ) => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = cache_config.num_gpu_blocks
                                * cache_config.block_size
                                * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    cache_config.num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn allocate_decoded_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        decoded_dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Option<Vec<DecodedKVCache>>> {
        if let Some(decoded_mb) = cache_config.cache_type.turboquant_cache_decoded_size() {
            tracing::info!("TurboQuant decoded block cache enabled: {} MB", decoded_mb);
        }

        let Some(decoded_cache_mb) = cache_config.cache_type.turboquant_cache_decoded_size() else {
            return Ok(None);
        };

        if !cache_config.cache_type.is_turboquant() {
            return Ok(None);
        }

        if !matches!(model_config.kv_cache_layout(), KvCacheLayout::Standard) {
            candle_core::bail!(
                "TurboQuant decoded block cache only supports standard KV cache layout."
            );
        }

        if decoded_cache_mb == 0 {
            return Ok(None);
        }

        let decoded_cache_bytes_total = decoded_cache_mb * 1024 * 1024;
        let num_layers = model_config.num_layers();

        if num_layers == 0 {
            return Ok(None);
        }

        // Simple first version: split decoded cache budget evenly across layers.
        let decoded_cache_bytes_per_layer = decoded_cache_bytes_total / num_layers;

        let mut decoded_layers = Vec::with_capacity(num_layers);

        for (layer_idx, device) in layer_devices
            .iter()
            .take(num_layers)
            .map(|x| x.as_ref().unwrap_or(device))
            .enumerate()
        {
            let block_size = cache_config.block_size;
            let num_kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
            let k_head_dim = model_config.k_head_dim_for_layer(layer_idx);
            let v_head_dim = model_config.v_head_dim_for_layer(layer_idx);

            let bytes_per_block = block_size
                * num_kv_heads
                * (k_head_dim + v_head_dim)
                * decoded_dtype.size_in_bytes();

            let mut num_decoded_blocks = decoded_cache_bytes_per_layer / bytes_per_block;

            // Make sure each layer has at least one slot if the user requested decoded caching.
            num_decoded_blocks = num_decoded_blocks.max(1);

            let key_cache = unsafe {
                Tensor::empty(
                    (num_decoded_blocks, num_kv_heads, block_size, k_head_dim),
                    decoded_dtype,
                    device,
                )?
            };

            let value_cache = unsafe {
                Tensor::empty(
                    (num_decoded_blocks, num_kv_heads, block_size, v_head_dim),
                    decoded_dtype,
                    device,
                )?
            };

            decoded_layers.push(DecodedKVCache {
                key_cache,
                value_cache,
                physical_to_slot: HashMap::new(),
                slot_to_physical: vec![None; num_decoded_blocks],
                lru_clock: vec![0; num_decoded_blocks],
                clock: 0,
                block_size,
                num_kv_heads,
                k_head_dim,
                v_head_dim,
                dtype: decoded_dtype,
            });
        }

        Ok(Some(decoded_layers))
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.k_head_dim_for_layer(layer_idx) / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.v_head_dim_for_layer(layer_idx),
            block_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::PagedCacheType;
    use candle_core::{DType, Result};

    #[test]
    fn parses_turboquant_cache_type() {
        assert_eq!(
            "turboquant".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::TurboQuant()
        );
        assert_eq!(
            "tq".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::TurboQuant()
        );
    }

    #[test]
    fn turboquant_cache_type_uses_u8_storage() -> Result<()> {
        assert_eq!(
            PagedCacheType::TurboQuant().to_dtype(DType::F16)?,
            DType::U8
        );
        Ok(())
    }
    #[test]
    fn parses_turboquant_cached_cache_type() {
        assert_eq!(
            "turboquant_cached:512".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::TurboQuantCached {
                decoded_cache_mb: 512
            }
        );
    }

    #[test]
    fn turboquant_cached_cache_type_uses_u8_primary_storage() -> Result<()> {
        assert_eq!(
            PagedCacheType::TurboQuantCached {
                decoded_cache_mb: 512
            }
            .to_dtype(DType::F16)?,
            DType::U8
        );
        Ok(())
    }

    #[test]
    fn parses_kvarn_cache_type() {
        assert_eq!(
            "kvarn".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::KVarN()
        );
        assert_eq!(
            "kvarn_k4v2_g128".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::KVarN()
        );
    }

    #[test]
    fn kvarn_cache_type_uses_u8_storage() -> Result<()> {
        assert_eq!(PagedCacheType::KVarN().to_dtype(DType::F16)?, DType::U8);
        Ok(())
    }
}
