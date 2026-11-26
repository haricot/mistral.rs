//! Exact typed sampling producer adapter. CUDA qualification is deferred to S10.
//! Each two-stage algorithm admits both plans before the first launch.
use crate::cuda::kernel_runtime::{kernel_cache, LaunchPlanCache};
use candle_core::{cuda_backend::cudarc::driver::{CudaStream, PushKernelArg}, DType, Device, Result};
use mistralrs_kernel_runtime::{
    model_scope::OperationRequirement,
    universal::{Access, Argument, ElementType, LaunchRequest, Parameter},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::{cell::RefCell, sync::Arc};

#[derive(Clone, Copy)]
enum ArgSpec { Tensor(&'static str, ElementType, Access), Scalar(&'static str) }
struct Spec {
    entry: &'static str, operation: &'static str, implementation: &'static str,
    block: u32, shared: u32, args: &'static [ArgSpec],
}
static SPECS: [Spec; 32] = [
    Spec { entry: "dkr_top1_stage1_single_f32", operation: "sampling.top1.stage1.single.v1", implementation: "cuda.dkr_top1_stage1_single_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::F32, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"),
    ] },
    Spec { entry: "dkr_top1_stage1_batch_f32", operation: "sampling.top1.stage1.batch.v1", implementation: "cuda.dkr_top1_stage1_batch_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::F32, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"),
    ] },
    Spec { entry: "dkr_top1_stage1_single_f16", operation: "sampling.top1.stage1.single.v1", implementation: "cuda.dkr_top1_stage1_single_f16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::F16, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"),
    ] },
    Spec { entry: "dkr_top1_stage1_batch_f16", operation: "sampling.top1.stage1.batch.v1", implementation: "cuda.dkr_top1_stage1_batch_f16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::F16, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"),
    ] },
    Spec { entry: "dkr_top1_stage1_single_bf16", operation: "sampling.top1.stage1.single.v1", implementation: "cuda.dkr_top1_stage1_single_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::Bf16, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"),
    ] },
    Spec { entry: "dkr_top1_stage1_batch_bf16", operation: "sampling.top1.stage1.batch.v1", implementation: "cuda.dkr_top1_stage1_batch_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::Bf16, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"),
    ] },
    Spec { entry: "dkr_top1_stage2_single_tokens", operation: "sampling.top1.stage2.single.tokens.v1", implementation: "cuda.dkr_top1_stage2_single_tokens.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("block_values", ElementType::F32, Access::Read), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Read), ArgSpec::Tensor("token_ids_out", ElementType::U32, Access::Write), ArgSpec::Scalar("nblocks"),
    ] },
    Spec { entry: "dkr_top1_stage2_single_both", operation: "sampling.top1.stage2.single.both.v1", implementation: "cuda.dkr_top1_stage2_single_both.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("block_values", ElementType::F32, Access::Read), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Read), ArgSpec::Tensor("packed_out", ElementType::F32, Access::Write), ArgSpec::Tensor("token_ids_out", ElementType::U32, Access::Write), ArgSpec::Scalar("nblocks"),
    ] },
    Spec { entry: "dkr_top1_stage2_batch_tokens", operation: "sampling.top1.stage2.batch.tokens.v1", implementation: "cuda.dkr_top1_stage2_batch_tokens.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("block_values", ElementType::F32, Access::Read), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Read), ArgSpec::Tensor("token_ids_out", ElementType::U32, Access::Write), ArgSpec::Scalar("nblocks"),
    ] },
    Spec { entry: "dkr_top1_stage2_batch_both", operation: "sampling.top1.stage2.batch.both.v1", implementation: "cuda.dkr_top1_stage2_batch_both.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("block_values", ElementType::F32, Access::Read), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Read), ArgSpec::Tensor("packed_out", ElementType::F32, Access::Write), ArgSpec::Tensor("token_ids_out", ElementType::U32, Access::Write), ArgSpec::Scalar("nblocks"),
    ] },
    Spec { entry: "dkr_top1_stage2_batch_packed", operation: "sampling.top1.stage2.batch.packed.v1", implementation: "cuda.dkr_top1_stage2_batch_packed.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("block_values", ElementType::F32, Access::Read), ArgSpec::Tensor("block_indices", ElementType::U32, Access::Read), ArgSpec::Tensor("packed_out", ElementType::F32, Access::Write), ArgSpec::Scalar("nblocks"),
    ] },
    Spec { entry: "dkr_categorical_stage1", operation: "sampling.categorical.stage1.v1", implementation: "cuda.dkr_categorical_stage1.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("input", ElementType::F32, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Write), ArgSpec::Tensor("block_sums", ElementType::F32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"), ArgSpec::Tensor("inv_temperatures", ElementType::F32, Access::Read),
    ] },
    Spec { entry: "dkr_categorical_stage2", operation: "sampling.categorical.stage2.v1", implementation: "cuda.dkr_categorical_stage2.v1", block: 256, shared: 16384, args: &[
        ArgSpec::Tensor("input", ElementType::F32, Access::Read), ArgSpec::Tensor("inv_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("block_values", ElementType::F32, Access::Read), ArgSpec::Tensor("block_sums", ElementType::F32, Access::Read), ArgSpec::Tensor("packed_out", ElementType::F32, Access::Write), ArgSpec::Scalar("ncols"), ArgSpec::Scalar("chunk_size"), ArgSpec::Scalar("nblocks"),
    ] },
    Spec { entry: "dkr_sample_ranked_topk", operation: "sampling.ranked.sample.v1", implementation: "cuda.dkr_sample_ranked_topk.v1", block: 128, shared: 0, args: &[
        ArgSpec::Tensor("packed", ElementType::F32, Access::Read), ArgSpec::Tensor("params", ElementType::F32, Access::Read), ArgSpec::Tensor("tokens", ElementType::U32, Access::Write), ArgSpec::Scalar("nrows"), ArgSpec::Scalar("packed_k"),
    ] },
    Spec { entry: "dkr_dflash_greedy_f32_f32_f32", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_f32_f32_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_f32_f32_bf16", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_f32_f32_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_f32_bf16_f32", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_f32_bf16_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_f32_bf16_bf16", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_f32_bf16_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_bf16_f32_f32", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_bf16_f32_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_bf16_f32_bf16", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_bf16_f32_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_bf16_bf16_f32", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_bf16_bf16_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_greedy_bf16_bf16_bf16", operation: "sampling.dflash.greedy.v1", implementation: "cuda.dkr_dflash_greedy_bf16_bf16_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_f32_f32_f32", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_f32_f32_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_f32_f32_bf16", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_f32_f32_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_f32_bf16_f32", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_f32_bf16_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_f32_bf16_bf16", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_f32_bf16_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::F32, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_bf16_f32_f32", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_bf16_f32_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_bf16_f32_bf16", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_bf16_f32_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_bf16_bf16_f32", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_bf16_bf16_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::F32, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_dflash_sample_bf16_bf16_bf16", operation: "sampling.dflash.sample.v1", implementation: "cuda.dkr_dflash_sample_bf16_bf16_bf16.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("packed_topk", ElementType::F32, Access::Read), ArgSpec::Tensor("projected_hidden", ElementType::Bf16, Access::Read), ArgSpec::Tensor("predecessor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("successor_codebook", ElementType::Bf16, Access::Read), ArgSpec::Tensor("anchors", ElementType::U32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("selected", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_ids", ElementType::U32, Access::Write), ArgSpec::Tensor("candidate_probs", ElementType::F32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("positions"), ArgSpec::Scalar("rank"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("k"), ArgSpec::Scalar("packed_width"),
    ] },
    Spec { entry: "dkr_sparse_rejection_categorical_f32", operation: "sampling.rejection.categorical.v1", implementation: "cuda.dkr_sparse_rejection_categorical_f32.v1", block: 256, shared: 0, args: &[
        ArgSpec::Tensor("target_logits", ElementType::F32, Access::Read), ArgSpec::Tensor("draft_tokens", ElementType::U32, Access::Read), ArgSpec::Tensor("q_token_ids", ElementType::U32, Access::Read), ArgSpec::Tensor("q_probs", ElementType::F32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("target_top_k", ElementType::U32, Access::Read), ArgSpec::Tensor("top_p", ElementType::F32, Access::Read), ArgSpec::Tensor("min_p", ElementType::F32, Access::Read), ArgSpec::Tensor("accept_uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("sample_uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("outcomes", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("drafts"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("q_width"),
    ] },
    Spec { entry: "dkr_sparse_rejection_topk_f32", operation: "sampling.rejection.topk.v1", implementation: "cuda.dkr_sparse_rejection_topk_f32.v1", block: 1, shared: 0, args: &[
        ArgSpec::Tensor("packed_target", ElementType::F32, Access::Read), ArgSpec::Tensor("draft_tokens", ElementType::U32, Access::Read), ArgSpec::Tensor("q_token_ids", ElementType::U32, Access::Read), ArgSpec::Tensor("q_probs", ElementType::F32, Access::Read), ArgSpec::Tensor("inverse_temperatures", ElementType::F32, Access::Read), ArgSpec::Tensor("target_top_k", ElementType::U32, Access::Read), ArgSpec::Tensor("top_p", ElementType::F32, Access::Read), ArgSpec::Tensor("min_p", ElementType::F32, Access::Read), ArgSpec::Tensor("accept_uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("sample_uniforms", ElementType::F32, Access::Read), ArgSpec::Tensor("outcomes", ElementType::U32, Access::Write), ArgSpec::Scalar("batch"), ArgSpec::Scalar("drafts"), ArgSpec::Scalar("vocab"), ArgSpec::Scalar("q_width"), ArgSpec::Scalar("packed_k"),
    ] },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DkrSamplingMode {
    Top1 { dtype: DType, batched: bool, packed: bool, tokens: bool },
    Categorical,
    Ranked,
    DFlash { hidden: DType, predecessor: DType, successor: DType, probabilistic: bool },
    Rejection { bounded: bool },
}
fn top1_dtype(dtype:DType)->Result<usize>{
    match dtype {DType::F32=>Ok(0),DType::F16=>Ok(1),DType::BF16=>Ok(2),_=>candle_core::bail!("sampling top1 dtype")}
}
fn dflash_dtype(dtype:DType)->Result<usize>{
    match dtype {DType::F32=>Ok(0),DType::BF16=>Ok(1),_=>candle_core::bail!("DFlash selector dtype")}
}
fn selector_dtype(code:i32)->Result<DType>{
    match code {0=>Ok(DType::F32),1=>Ok(DType::BF16),_=>candle_core::bail!("DFlash selector dtype code")}
}
fn indices(mode:DkrSamplingMode)->Result<(usize,Option<usize>)>{
    Ok(match mode{
        DkrSamplingMode::Top1{dtype,batched,packed,tokens}=>{
            let second=match (batched,packed,tokens){
                (false,false,true)=>6,(false,true,true)=>7,
                (true,false,true)=>8,(true,true,true)=>9,(true,true,false)=>10,
                _=>candle_core::bail!("invalid top1 output mode"),
            };
            (top1_dtype(dtype)?*2+(batched as usize),Some(second))
        }
        DkrSamplingMode::Categorical=>(11,Some(12)),
        DkrSamplingMode::Ranked=>(13,None),
        DkrSamplingMode::DFlash{hidden,predecessor,successor,probabilistic}=>(
            14+(probabilistic as usize)*8+dflash_dtype(hidden)?*4+dflash_dtype(predecessor)?*2+dflash_dtype(successor)?,None),
        DkrSamplingMode::Rejection{bounded}=>(30+(bounded as usize),None),
    })
}
/// Exact capability producer for the selected request route. The model
/// admission layer must union all reachable modes and dtype combinations.
pub fn dkr_sampling_requirements(mode:DkrSamplingMode)->Result<Vec<OperationRequirement>>{
    let (a,b)=indices(mode)?;
    Ok(std::iter::once(a).chain(b).map(|id|OperationRequirement{
        operation:SPECS[id].operation.into(),implementations:Some(vec![SPECS[id].implementation.into()])
    }).collect())
}
#[derive(Clone,Copy)]
enum Binding { Tensor(u64,usize), Scalar(i32) }
use Binding::{Tensor as T,Scalar as S};
#[derive(Clone,Copy)]
struct Bindings { values:[Binding;16], len:usize }
impl Bindings{
    fn new()->Self{Self{values:[S(0);16],len:0}}
    fn push(&mut self,v:Binding){self.values[self.len]=v;self.len+=1;}
    fn extend(&mut self,values:&[Binding]){for v in values{self.push(*v);}}
    fn as_slice(&self)->&[Binding]{&self.values[..self.len]}
}
#[derive(Clone,Copy,Eq,PartialEq)]
struct StageKey{index:usize,grid:[u32;3],signature:[usize;16]}
#[derive(Clone,Copy,Eq,PartialEq)]
struct Key{first:StageKey,second:Option<StageKey>}
struct Stage { key:StageKey, bindings:Bindings }
struct Plan { first:PrevalidatedLaunchPlan, second:Option<PrevalidatedLaunchPlan> }
thread_local!{static PLANS:RefCell<LaunchPlanCache<Key,Plan>>=RefCell::new(LaunchPlanCache::new());}
fn err(e:impl std::fmt::Display)->candle_core::Error{candle_core::Error::msg(e.to_string())}
fn positive(v:i32)->Result<usize>{
    if v<=0{candle_core::bail!("non-positive sampling geometry");}
    Ok(v as usize)
}
fn mul(a:usize,b:usize)->Result<usize>{a.checked_mul(b).ok_or_else(||err("sampling span overflow"))}
impl Stage{
    fn new(index:usize,grid:[u32;3],values:&[Binding])->Result<Self>{
        let spec=&SPECS[index];
        if grid.contains(&0)||values.len()!=spec.args.len(){candle_core::bail!("sampling binding geometry");}
        let mut signature=[0;16];
        for (i,(arg,binding)) in spec.args.iter().zip(values).enumerate(){
            signature[i]=match (arg,binding){
                (ArgSpec::Tensor(_,dt,_),T(ptr,count)) if *ptr!=0 && *count!=0 && *ptr%(dt.size_bytes() as u64)==0=>{
                    let bytes=mul(*count,dt.size_bytes())?;
                    if ptr.checked_add(u64::try_from(bytes).map_err(err)?).is_none(){candle_core::bail!("sampling address overflow");}
                    *count
                },
                (ArgSpec::Scalar(_),S(value))=>positive(*value)?,
                _=>candle_core::bail!("sampling pointer or ABI binding mismatch"),
            };
        }
        let mut bindings=Bindings::new();bindings.extend(values);
        Ok(Self{key:StageKey{index,grid,signature},bindings})
    }
    fn prepare(&self,device:&Device)->Result<PrevalidatedLaunchPlan>{
        let spec=&SPECS[self.key.index];
        let args=spec.args.iter().zip(self.bindings.as_slice()).map(|(arg,binding)|match (arg,binding){
            (ArgSpec::Tensor(_,dt,_),T(_,count))=>Argument::Tensor{dtype:*dt,shape:vec![*count],contiguous:true,start_offset:0,representation:None},
            (ArgSpec::Scalar(_),S(_))=>Argument::Scalar{dtype:ElementType::I32},
            _=>unreachable!("validated sampling binding"),
        }).collect();
        let mut request=LaunchRequest{operation:spec.operation.into(),cc:0,args,grid:self.key.grid,max_dynamic_shared_mem_bytes:spec.shared};
        let p=kernel_cache()?.prepare_for_device(&mut request,device).map_err(err)?;
        let m=&p.manifest;
        if m.entry!=spec.entry||m.implementation_id!=spec.implementation||m.block!=[spec.block,1,1]||
            m.shared_mem_bytes!=spec.shared||m.arguments.len()!=spec.args.len()||
            !m.arguments.iter().zip(spec.args).all(|(actual,expected)|match (actual,expected){
                (Parameter::Tensor{name,dtype,rank,access,representation},ArgSpec::Tensor(n,dt,a))=>
                    name.as_str()==*n&&dtype==dt&&*rank==1&&access==a&&representation.is_none(),
                (Parameter::Scalar{name,dtype},ArgSpec::Scalar(n))=>name.as_str()==*n&&*dtype==ElementType::I32,
                _=>false,
            }){
            candle_core::bail!("sampling exact profile/ABI mismatch");
        }
        unsafe{p.prevalidate_launch_with_default_shared(&request,device,spec.shared)}.map_err(err)
    }
    unsafe fn submit(&self,plan:&PrevalidatedLaunchPlan,stream:&CudaStream)->Result<()>{
        unsafe{plan.with_prevalidated_launch(|func,cfg|{
            let mut b=stream.launch_builder(func);
            for value in self.bindings.as_slice(){match value{T(ptr,_)=>{b.arg(ptr);},S(v)=>{b.arg(v);}}}
            b.launch(cfg)?;Ok(())
        })}.map_err(err)
    }
}
/// Callers retain the actual CUDA storage and stream guards for every span.
unsafe fn launch(device:&Device,stream:&CudaStream,first:Stage,second:Option<Stage>)->Result<()>{
    let expected=device.as_cuda_device()?.cuda_stream();
    if !Arc::ptr_eq(expected.context(),stream.context())||expected.cu_stream()!=stream.cu_stream(){
        candle_core::bail!("sampling stream/context mismatch");
    }
    let key=Key{first:first.key,second:second.as_ref().map(|s|s.key)};
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,device,||{
        Ok(Plan{first:first.prepare(device)?,second:second.as_ref().map(|s|s.prepare(device)).transpose()?})
    },|p|{
        unsafe{first.submit(&p.first,stream)?;}
        if let (Some(s),Some(plan))=(&second,&p.second){unsafe{s.submit(plan,stream)?;}}
        Ok(())
    }))
}
fn geometry(rows:i32,cols:i32,chunk:i32,blocks:i32)->Result<(usize,usize,usize)>{
    let r=positive(rows)?;let c=positive(cols)?;let b=positive(blocks)?;
    if r>65535||chunk!=2048||b!=c.div_ceil(2048)||c>(1usize<<24)+1{
        candle_core::bail!("invalid sampling vocabulary geometry");
    }
    Ok((r,mul(r,c)?,mul(r,b)?))
}
/// Safety: pointers cover the validated contiguous spans and writable spans
/// do not alias. Null optional outputs are excluded from the typed ABI.
pub(crate) unsafe fn launch_top1(device:&Device,stream:&CudaStream,dtype:DType,batched:bool,
    rows:i32,cols:i32,chunk:i32,blocks:i32,input:u64,values:u64,ids:u64,packed:u64,tokens:u64)->Result<()>{
    let (r,input_len,work)=geometry(rows,cols,chunk,blocks)?;
    if !batched&&r!=1{candle_core::bail!("single top1 row count");}
    let (a,b)=indices(DkrSamplingMode::Top1{dtype,batched,packed:packed!=0,tokens:tokens!=0})?;
    let first=Stage::new(a,[blocks as u32,rows as u32,1],&[T(input,input_len),T(values,work),T(ids,work),S(cols),S(chunk)])?;
    let mut args=Bindings::new();args.extend(&[T(values,work),T(ids,work)]);
    if packed!=0{args.push(T(packed,mul(r,2)?));}if tokens!=0{args.push(T(tokens,r));}args.push(S(blocks));
    let second=Stage::new(b.expect("top1 second stage"),[rows as u32,1,1],args.as_slice())?;
    unsafe{launch(device,stream,first,Some(second))}
}
pub(crate) unsafe fn launch_categorical(device:&Device,stream:&CudaStream,rows:i32,cols:i32,chunk:i32,blocks:i32,
    input:u64,temperatures:u64,uniforms:u64,values:u64,sums:u64,packed:u64)->Result<()>{
    let (r,input_len,work)=geometry(rows,cols,chunk,blocks)?;
    let first=Stage::new(11,[blocks as u32,rows as u32,1],&[T(input,input_len),T(values,work),T(sums,work),S(cols),S(chunk),T(temperatures,r)])?;
    let second=Stage::new(12,[rows as u32,1,1],&[T(input,input_len),T(temperatures,r),T(uniforms,r),T(values,work),T(sums,work),T(packed,mul(r,2)?),S(cols),S(chunk),S(blocks)])?;
    unsafe{launch(device,stream,first,Some(second))}
}
pub(crate) unsafe fn launch_ranked(device:&Device,stream:&CudaStream,rows:i32,k:i32,packed:u64,params:u64,tokens:u64)->Result<()>{
    let r=positive(rows)?;let width=positive(k)?;
    if width>128{candle_core::bail!("ranked sampling k");}
    let first=Stage::new(13,[u32::try_from(r.div_ceil(128)).map_err(err)?,1,1],&[T(packed,mul(mul(r,width)?,2)?),T(params,mul(r,5)?),T(tokens,r),S(rows),S(k)])?;
    unsafe{launch(device,stream,first,None)}
}
pub(crate) unsafe fn launch_dflash(device:&Device,stream:&CudaStream,types:[i32;3],dims:[i32;6],
    packed:u64,hidden:u64,predecessor:u64,successor:u64,anchors:u64,selected:u64,
    sampling:Option<[u64;4]>)->Result<()>{
    let [batch,positions,rank,vocab,k,width]=dims;
    let r=mul(positive(batch)?,positive(positions)?)?;
    let rank_len=positive(rank)?;let vocab_len=positive(vocab)?;let k_len=positive(k)?;
    if r>i32::MAX as usize||k_len>128||width!=2*k||vocab_len>(1usize<<24)+1{candle_core::bail!("DFlash geometry");}
    let [h,p,s]=types;
    let (id,_)=indices(DkrSamplingMode::DFlash{hidden:selector_dtype(h)?,predecessor:selector_dtype(p)?,successor:selector_dtype(s)?,probabilistic:sampling.is_some()})?;
    let book=mul(vocab_len,rank_len)?;
    let mut args=Bindings::new();
    args.extend(&[T(packed,mul(r,positive(width)?)?),T(hidden,mul(r,rank_len)?),T(predecessor,book),T(successor,book),T(anchors,batch as usize)]);
    if let Some([temperature,uniform,_,_])=sampling{args.extend(&[T(temperature,batch as usize),T(uniform,r)]);}
    args.push(T(selected,r));
    if let Some([_,_,candidate_ids,candidate_probs])=sampling{
        let n=mul(r,k_len)?;args.extend(&[T(candidate_ids,n),T(candidate_probs,n)]);
    }
    for dim in dims{args.push(S(dim));}
    let first=Stage::new(id,[batch as u32,1,1],args.as_slice())?;
    unsafe{launch(device,stream,first,None)}
}
pub(crate) unsafe fn launch_rejection(device:&Device,stream:&CudaStream,pointers:[u64;11],
    batch:i32,drafts:i32,vocab:i32,q_width:i32,packed_k:Option<i32>)->Result<()>{
    let b=positive(batch)?;let d=positive(drafts)?;let v=positive(vocab)?;let q=positive(q_width)?;
    if q>128||drafts==i32::MAX{candle_core::bail!("rejection geometry");}
    let draft_count=mul(b,d)?;let q_count=mul(draft_count,q)?;
    let target_width=if let Some(k)=packed_k{let k=positive(k)?;if k>128{candle_core::bail!("rejection packed k");}mul(k,2)?}else{v};
    let counts=[mul(mul(b,d+1)?,target_width)?,draft_count,q_count,q_count,b,b,b,b,draft_count,b,mul(b,3)?];
    let mut args=Bindings::new();for (ptr,count) in pointers.into_iter().zip(counts){args.push(T(ptr,count));}
    args.extend(&[S(batch),S(drafts),S(vocab),S(q_width)]);
    if let Some(k)=packed_k{args.push(S(k));}
    let first=Stage::new(30+(packed_k.is_some() as usize),[batch as u32,1,1],args.as_slice())?;
    unsafe{launch(device,stream,first,None)}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_sampling_capabilities(){
        let mut entries=std::collections::BTreeSet::new();
        for dt in [DType::F32,DType::F16,DType::BF16]{
            for batched in [false,true]{
                for (packed,tokens) in [(false,true),(true,true),(true,false)]{
                    if !batched&&!tokens{continue;}
                    for r in dkr_sampling_requirements(DkrSamplingMode::Top1{dtype:dt,batched,packed,tokens}).unwrap(){entries.extend(r.implementations.unwrap());}
                }
            }
        }
        for h in [DType::F32,DType::BF16]{for p in [DType::F32,DType::BF16]{for s in [DType::F32,DType::BF16]{
            for probabilistic in [false,true]{for r in dkr_sampling_requirements(DkrSamplingMode::DFlash{hidden:h,predecessor:p,successor:s,probabilistic}).unwrap(){entries.extend(r.implementations.unwrap());}}
        }}}
        for mode in [DkrSamplingMode::Categorical,DkrSamplingMode::Ranked,DkrSamplingMode::Rejection{bounded:false},DkrSamplingMode::Rejection{bounded:true}]{
            for r in dkr_sampling_requirements(mode).unwrap(){entries.extend(r.implementations.unwrap());}
        }
        assert_eq!(entries.len(),32);
        assert!(dkr_sampling_requirements(DkrSamplingMode::Top1{dtype:DType::F32,batched:false,packed:true,tokens:false}).is_err());
    }
    #[test]
    fn dkr_sampling_binding_geometry(){
        assert_eq!(geometry(2,2049,2048,2).unwrap(),(2,4098,4));
        assert!(geometry(2,2049,2048,1).is_err());
        assert!(Stage::new(13,[1,1,1],&[T(4,4),T(8,5),T(12,1),S(1),S(2)]).is_ok());
        assert!(Stage::new(13,[1,1,1],&[T(3,4),T(8,5),T(12,1),S(1),S(2)]).is_err());
        assert!(Stage::new(13,[1,1,1],&[T(4,4),T(8,5),T(0,1),S(1),S(2)]).is_err());
    }
}
