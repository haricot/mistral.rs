use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::{
    config::{KvCacheLayout, ModelConfigLike},
    turboquant_cache,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    #[serde(alias = "auto")]
    #[default]
    Auto,
    #[serde(alias = "f8e4m3")]
    F8E4M3,
    #[serde(alias = "turboquant", alias = "tq")]
    TurboQuant,
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> Result<DType> {
        match self {
            PagedCacheType::F8E4M3 => Ok(DType::F8E4M3),
            PagedCacheType::Auto => Ok(act_dtype),
            PagedCacheType::TurboQuant => Ok(DType::U8),
        }
    }

    pub fn is_turboquant(&self) -> bool {
        matches!(self, PagedCacheType::TurboQuant)
    }

    pub fn bytes_per_block_all_layers(
        &self,
        act_dtype: DType,
        model_config: &dyn ModelConfigLike,
        block_size: usize,
    ) -> Result<usize> {
        match (self, model_config.kv_cache_layout()) {
            (PagedCacheType::TurboQuant, KvCacheLayout::Standard) => {
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
            (PagedCacheType::TurboQuant, KvCacheLayout::Mla { .. }) => {
                candle_core::bail!("TurboQuant paged KV cache does not support MLA cache layout.")
            }
            (_, KvCacheLayout::Standard) => {
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
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            "turboquant" | "tq" => Ok(Self::TurboQuant),
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected `auto`, `f8e4m3`, or `turboquant`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let dtype = cache_config.cache_type.to_dtype(dtype)?;
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                layer_devices,
            )?)),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
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
                (PagedCacheType::TurboQuant, KvCacheLayout::Standard) => {
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
                (PagedCacheType::TurboQuant, KvCacheLayout::Mla { .. }) => {
                    candle_core::bail!(
                        "TurboQuant paged KV cache does not support MLA cache layout."
                    )
                }
                (_, KvCacheLayout::Standard) => {
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
            PagedCacheType::TurboQuant
        );
        assert_eq!(
            "tq".parse::<PagedCacheType>().unwrap(),
            PagedCacheType::TurboQuant
        );
    }

    #[test]
    fn turboquant_cache_type_uses_u8_storage() -> Result<()> {
        assert_eq!(PagedCacheType::TurboQuant.to_dtype(DType::F16)?, DType::U8);
        Ok(())
    }
}
