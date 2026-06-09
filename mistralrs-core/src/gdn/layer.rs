use candle_core::{DType, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::pipeline::RecurrentBatchKind;

use super::backend;
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::GdnProjection;
use super::weights::{GdnWeightLoadCtx, GdnWeightMode, GdnWeights};

fn gdn_finite_check_enabled() -> bool {
    std::env::var("MISTRALRS_QWEN35_MOE_FINITE_CHECK")
        .is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
}

fn ensure_gdn_finite(xs: &Tensor, label: impl AsRef<str>) -> Result<()> {
    if !gdn_finite_check_enabled() {
        return Ok(());
    }

    let max_abs = xs
        .to_dtype(DType::F32)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    if !max_abs.is_finite() {
        candle_core::bail!(
            "GDN non-finite activation at {}: dtype={:?}, shape={:?}, max_abs={max_abs}",
            label.as_ref(),
            xs.dtype(),
            xs.dims()
        );
    }
    Ok(())
}

pub struct GatedDeltaNet {
    pub in_proj: Arc<dyn QuantMethod>,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    dims: GdnDims,
}

impl GatedDeltaNet {
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &dyn GdnConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<Comm>,
        weight_mode: GdnWeightMode,
    ) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let weights = GdnWeights::load(
            vb,
            GdnWeightLoadCtx {
                cfg,
                dims: &dims,
                mapper,
                layer_idx,
                loading_isq,
                comm,
                weight_mode,
            },
        )?;
        Ok(Self {
            in_proj: weights.in_proj,
            conv1d_weight: weights.conv1d_weight,
            dt_bias: weights.dt_bias,
            a_log: weights.a_log,
            norm: weights.norm,
            out_proj: weights.out_proj,
            dims,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        let projected = self.project(x, batch_size, seq_len)?;
        ensure_gdn_finite(&projected.q, "project.q")?;
        ensure_gdn_finite(&projected.k, "project.k")?;
        ensure_gdn_finite(&projected.v, "project.v")?;
        ensure_gdn_finite(&projected.z, "project.z")?;
        ensure_gdn_finite(&projected.b, "project.b")?;
        ensure_gdn_finite(&projected.a, "project.a")?;
        let mixed_qkv = projected.conv_input(&self.dims, batch_size, seq_len)?;
        ensure_gdn_finite(&mixed_qkv, "conv.input")?;
        let mixed_qkv = backend::causal_conv1d(
            &mixed_qkv,
            &self.conv1d_weight,
            &self.dims,
            cache,
            batch_kind,
        )?;
        ensure_gdn_finite(&mixed_qkv, "conv.output")?;
        let y = backend::apply_recurrence_from_convolved(
            &mixed_qkv,
            &projected.b,
            &projected.a,
            &self.a_log,
            &self.dt_bias,
            &self.dims,
            batch_size,
            seq_len,
            cache,
            dtype,
        )?;
        ensure_gdn_finite(&y, "recurrence.output")?;

        self.finish_forward(y, projected.z, batch_size, seq_len, dtype)
    }

    fn project(&self, x: &Tensor, batch_size: usize, seq_len: usize) -> Result<GdnProjection> {
        let mixed = self.in_proj.forward(x)?;
        GdnProjection::from_packed(mixed, &self.dims, batch_size, seq_len)
    }

    pub fn residual_input_projection_tensors(&self) -> (Tensor, Tensor) {
        let weight = self
            .in_proj
            .dequantize_w()
            .expect("failed to dequantize GDN input projection");
        let qkvz = weight
            .narrow(0, 0, self.dims.qkvz_out_dim())
            .expect("failed to split GDN qkvz projection");
        let ba = weight
            .narrow(0, self.dims.qkvz_out_dim(), self.dims.ba_out_dim())
            .expect("failed to split GDN ba projection");
        (qkvz, ba)
    }

    fn finish_forward(
        &self,
        y: Tensor,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
        _dtype: DType,
    ) -> Result<Tensor> {
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.dims.head_v_dim))?;
        let z = z.reshape(((), self.dims.head_v_dim))?;
        ensure_gdn_finite(&y, "finish.pre_norm_y")?;
        ensure_gdn_finite(&z, "finish.pre_norm_z")?;
        let y = self.norm.forward(&y, &z)?;
        ensure_gdn_finite(&y, "finish.norm")?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.dims.value_dim))?;
        ensure_gdn_finite(&y, "finish.pre_out_proj")?;
        let y = self.out_proj.forward(&y)?;
        ensure_gdn_finite(&y, "finish.out_proj")?;
        Ok(y)
    }
}
