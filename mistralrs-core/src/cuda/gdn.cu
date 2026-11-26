#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include "gdn_recurrence_conv_device.cuh"

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
using gdn_fp8_e4m3 = __nv_fp8_e4m3;
#else
struct alignas(1) gdn_fp8_e4m3 {
  unsigned char value;

  __device__ explicit gdn_fp8_e4m3(float) : value(0) {}
};
#endif

constexpr int GDN_CHANNEL_BLOCK_SIZE = 256;
#include "gdn_decode_device.cuh"
constexpr int GDN_PREFILL_CONV_THREADS = 128;
constexpr int GDN_PREFILL_CONV_TOKEN_TILE = 64;
constexpr int GDN_STATE_DTYPE_F16 = 0;
constexpr int GDN_STATE_DTYPE_BF16 = 1;
constexpr int GDN_STATE_DTYPE_F32 = 2;
constexpr uint32_t GDN_PENDING_KEY_BANK_MASK = 1u;




#include "gdn_preparation_device.cuh"

extern "C" void gdn_packed_to_padded(
    const void *source, void *output, const uint32_t *cu_seqlens,
    int batch_size, int padded_len, int width, int64_t source_token_stride,
    float padding_value, int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t total_elements =
      (size_t)batch_size * padded_len * width;
  const int blocks =
      min((size_t)65535, (total_elements + THREADS - 1) / THREADS);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_packed_to_padded_kernel<<<blocks, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)source, (__nv_bfloat16 *)output, cu_seqlens,
        total_elements, padded_len, width, source_token_stride, padding_value);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_packed_to_padded_kernel<<<blocks, THREADS, 0, custream>>>(
        (const float *)source, (float *)output, cu_seqlens, total_elements,
        padded_len, width, source_token_stride, padding_value);
  }
}

extern "C" void gdn_padded_to_packed(
    const void *source, void *output, const uint32_t *cu_seqlens,
    int batch_size, int padded_len, int width, int64_t source_batch_stride,
    int64_t source_token_stride, int64_t source_feature_stride,
    int feature_inner_width, int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t max_row_elements = (size_t)padded_len * width;
  const int row_blocks =
      min((size_t)65535, (max_row_elements + THREADS - 1) / THREADS);
  const dim3 grid(batch_size, row_blocks);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_padded_to_packed_kernel<<<grid, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)source, (__nv_bfloat16 *)output, cu_seqlens,
        width, source_batch_stride, source_token_stride, source_feature_stride,
        feature_inner_width);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_padded_to_packed_kernel<<<grid, THREADS, 0, custream>>>(
        (const float *)source, (float *)output, cu_seqlens, width,
        source_batch_stride, source_token_stride, source_feature_stride,
        feature_inner_width);
  }
}

extern "C" void gdn_extract_ragged_conv_state(
    const void *padded_input, const void *initial_state, void *output,
    const uint32_t *cu_seqlens, int batch_size, int padded_len, int channels,
    int state_width, int64_t input_batch_stride, int64_t input_token_stride,
    int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t total_elements =
      (size_t)batch_size * channels * state_width;
  const int blocks =
      min((size_t)65535, (total_elements + THREADS - 1) / THREADS);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_extract_ragged_conv_state_kernel<<<blocks, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)padded_input,
        (const __nv_bfloat16 *)initial_state, (__nv_bfloat16 *)output,
        cu_seqlens, total_elements, padded_len, channels, state_width,
        input_batch_stride, input_token_stride);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_extract_ragged_conv_state_kernel<<<blocks, THREADS, 0, custream>>>(
        (const float *)padded_input, (const float *)initial_state,
        (float *)output, cu_seqlens, total_elements, padded_len, channels,
        state_width, input_batch_stride, input_token_stride);
  }
}

// (batch, head) -> row of the state buffer: identity on a gathered [B*H, ...] copy, or through the
// per-batch slot table when a kernel updates the recurrent state pool in place

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence (optimized)
//
// V-tiled recurrence with compile-time K dimension for register residency.
// Grid: (ceil(V/BV), B*H), Block: (BV,). Each thread owns BK registers of
// state. Shared memory holds k_buf and q_buf (2*BK floats).
//
// Optimizations over naive version:
//   - Template BK -> float s[BK] lives in true registers (1 cycle vs ~30)
//   - #pragma unroll on all k-loops -> full ILP
//   - Fused decay+kv_mem pass and fused state_update+output pass
//   - __fmaf_rn intrinsics for guaranteed fused multiply-add
//   - BV=64 threads -> 2 warps, 6 blocks/SM on Ampere
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] (in/out)  output: [BH, S, V]
// ============================================================================

// Optimized kernel: BK known at compile time -> registers + full unrolling

// Fallback kernel: runtime k_dim, still V-tiled for occupancy

template <typename StateT>
void launch_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim == 128) {
    // Fast path for Qwen3-Next (k_dim=128)
    constexpr int BK = 128;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<StateT, BK, BV>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    // Fast path for models with k_dim=64
    constexpr int BK = 64;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<StateT, BK, BV>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else {
    // Fallback for other k_dim values (runtime loop, still V-tiled)
    constexpr int BV = 64;
    constexpr int MAX_K = 256;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    size_t smem = 2 * k_dim * sizeof(float);
    gated_delta_rule_recurrence_kernel_fallback<StateT, BV, MAX_K>
        <<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                        seq_len, k_dim, v_dim, slot_indices,
                                        num_heads);
  }
}

extern "C" void gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}


template <typename StateT>
void launch_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);

  if (k_dim == 128) {
    gated_delta_rule_recurrence_kernel_warp<StateT, 128, NUM_WARPS>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    gated_delta_rule_recurrence_kernel_warp<StateT, 64, NUM_WARPS>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else {
    launch_gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh,
                                       seq_len, k_dim, v_dim, slot_indices,
                                       num_heads, stream);
  }
}

extern "C" void warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT>
void launch_vmajor_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return;
  }

  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);
  gated_delta_rule_recurrence_kernel_warp<StateT, 128, NUM_WARPS, true>
      <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                   v_dim, slot_indices, num_heads);
}

extern "C" void vmajor_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}


template <typename StateT, int VALUES_PER_WARP>
cudaError_t launch_vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return cudaErrorInvalidValue;
  }

  constexpr int NUM_WARPS = 4;
  constexpr int VALUES_PER_BLOCK = NUM_WARPS * VALUES_PER_WARP;
  const dim3 grid((v_dim + VALUES_PER_BLOCK - 1) / VALUES_PER_BLOCK, bh);
  const dim3 block(32, NUM_WARPS);
  gated_delta_rule_recurrence_kernel_vmajor_grouped<StateT, VALUES_PER_WARP,
                                                     NUM_WARPS>
      <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                  slot_indices, num_heads);
  return cudaGetLastError();
}

template <typename StateT>
cudaError_t dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int values_per_warp, cudaStream_t stream) {
  switch (values_per_warp) {
  case 2:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 2>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  case 4:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 4>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  case 8:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 8>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

extern "C" int vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int values_per_warp, int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim,
            v_dim, slot_indices, num_heads, values_per_warp, custream));
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len,
            k_dim, v_dim, slot_indices, num_heads, values_per_warp, custream));
  } else {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim,
            v_dim, slot_indices, num_heads, values_per_warp, custream));
  }
}


// ============================================================================
// Kernel 1b: chunked_gated_delta_rule_recurrence (prefill optimization)
//
// Processes prefill tokens in BT-token chunks instead of one at a time.
// Within each chunk: parallel prefix sum of g, cooperative kk_dot computation,
// forward substitution (triangular solve), output computation, and state
// update.
//
// Same thread model as Kernel 1: one block per (v_tile, batch*head),
// one thread per V-column. Each thread owns BK registers of state.
//
// Shared memory holds:
//   k_chunk[BT * BK]  -- key vectors for current chunk
//   kk_dot[BT * BT]   -- dot(k[i], k[j]) lower-triangular matrix
//   gcum[BT]           -- cumulative sum of g within chunk
//   beta_s[BT]         -- beta values for chunk
//   q_buf[BK]          -- q vector (loaded one row at a time)
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] or [BH, V, K] (in/out)  output: [BH, S, V]
// ============================================================================


// Same device-resource policy as the typed catalogue, without a CC test.
template <typename StateT, int BT, int BK, bool VALUE_MAJOR>
cudaError_t launch_gdn_chunk_variant(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int v_dim, const int32_t *slots, int heads, cudaStream_t stream,
    int normal, int capacity) {
  constexpr int BV = 64;
  constexpr int smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);
  if (smem > capacity) return cudaErrorInvalidValue;
  auto kernel = chunked_gated_delta_rule_kernel<StateT, BT, BK, BV, VALUE_MAJOR>;
  if (smem > normal) {
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (status != cudaSuccess) return status;
  }
  dim3 grid((v_dim + BV - 1) / BV, bh);
  dim3 block(BV);
  kernel<<<grid, block, smem, stream>>>(q,k,v,g,beta,state,output,seq_len,v_dim,slots,heads);
  return cudaGetLastError();
}

template <typename StateT, int BK, bool VALUE_MAJOR>
cudaError_t dispatch_gdn_chunk_resource(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int v_dim, const int32_t *slots, int heads, cudaStream_t stream) {
  int device = 0, normal = 0, optin = 0;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&normal, cudaDevAttrMaxSharedMemoryPerBlock, device);
  if (status != cudaSuccess) return status;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess)
    optin = normal;
  const int capacity = normal > optin ? normal : optin;
  constexpr int large = (64 * BK + 64 * 64 + 128 + BK) * sizeof(float);
  if (capacity >= large)
    return launch_gdn_chunk_variant<StateT,64,BK,VALUE_MAJOR>(q,k,v,g,beta,state,output,bh,seq_len,v_dim,slots,heads,stream,normal,capacity);
  return launch_gdn_chunk_variant<StateT,32,BK,VALUE_MAJOR>(q,k,v,g,beta,state,output,bh,seq_len,v_dim,slots,heads,stream,normal,capacity);
}

template <typename StateT, bool VALUE_MAJOR = false>
void launch_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim == 128) {
    dispatch_gdn_chunk_resource<StateT,128,VALUE_MAJOR>(q,k,v,g,beta,state,output,bh,seq_len,v_dim,slot_indices,num_heads,stream);
  } else if (k_dim == 64) {
    dispatch_gdn_chunk_resource<StateT,64,VALUE_MAJOR>(q,k,v,g,beta,state,output,bh,seq_len,v_dim,slot_indices,num_heads,stream);
  } else if constexpr (!VALUE_MAJOR) {
    launch_gated_delta_rule_recurrence(q,k,v,g,beta,state,output,bh,seq_len,k_dim,v_dim,slot_indices,num_heads,stream);
  }
}

extern "C" void chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT>
cudaError_t launch_vmajor_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) return cudaErrorInvalidValue;
  return dispatch_gdn_chunk_resource<StateT,128,true>(q,k,v,g,beta,state,output,bh,seq_len,v_dim,slot_indices,num_heads,stream);
}

extern "C" int vmajor_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream));
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream));
  } else {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream));
  }
}

// ============================================================================
// Kernel 2a: causal_conv1d_update (decode path, single step)
//
// Each thread handles one channel: shift conv_state left by 1,
// insert new value, dot product with weight, apply SiLU.
//
// x: [B, 1, conv_dim]  weight: [conv_dim, kernel_size]
// conv_state: [B, conv_dim, kernel_size] (in/out)
// output: [B, 1, conv_dim]
// ============================================================================


extern "C" void causal_conv1d_update(const void *x, const void *weight,
                                     void *conv_state, void *output,
                                     int batch_size, int conv_dim,
                                     int kernel_size, int64_t x_stride_b,
                                     int64_t x_stride_s, int64_t x_stride_c,
                                     const int32_t *slot_indices, int dtype,
                                     int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(GDN_CHANNEL_BLOCK_SIZE);
  dim3 grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE,
            batch_size);

  if (kernel_size == GDN_PACKED_CONV_WIDTH) {
    if (dtype == 0) {
      causal_conv1d_update_width4_kernel<__half><<<grid, block, 0, custream>>>(
          (const __half *)x, (const __half *)weight, (__half *)conv_state,
          (__half *)output, batch_size, conv_dim, x_stride_b, x_stride_s,
          x_stride_c, slot_indices);
    } else {
      causal_conv1d_update_width4_kernel<__nv_bfloat16>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
              (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
              conv_dim, x_stride_b, x_stride_s, x_stride_c, slot_indices);
    }
  } else if (dtype == 0) {
    causal_conv1d_update_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)weight, (__half *)conv_state,
        (__half *)output, batch_size, conv_dim, kernel_size, x_stride_b,
        x_stride_s, x_stride_c, slot_indices);
  } else {
    causal_conv1d_update_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
        conv_dim, kernel_size, x_stride_b, x_stride_s, x_stride_c,
        slot_indices);
  }
}

// ============================================================================
// Kernel 2b: causal_conv1d_full (prefill path)
//
// Each thread handles one (channel, position), seeded from the prior state.
// A second pass retains the last kernel_size positions.
//
// x: [B, S, conv_dim]  weight: [conv_dim, kernel_size]
// conv_state_out: [B, conv_dim, kernel_size]  output: [B, S, conv_dim]
// ============================================================================


extern "C" void causal_conv1d_full(const void *x, const void *weight,
                                   const void *conv_state_in,
                                   void *conv_state_out, void *output,
                                   int batch_size, int conv_dim, int seq_len,
                                   int kernel_size, int64_t x_stride_b,
                                   int64_t x_stride_s, int64_t x_stride_c,
                                   const int32_t *slot_indices, int dtype,
                                   int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  const dim3 state_block(256);
  dim3 block = state_block;
  const size_t plane = (size_t)conv_dim * seq_len;
  dim3 grid((unsigned int)((plane + 255) / 256), batch_size);

  const bool use_width4_tiled =
      kernel_size == GDN_PACKED_CONV_WIDTH && x_stride_c == 1;
  if (use_width4_tiled) {
    block = dim3(GDN_PREFILL_CONV_THREADS);
    grid = dim3((conv_dim + GDN_PREFILL_CONV_THREADS - 1) /
                    GDN_PREFILL_CONV_THREADS,
                (seq_len + GDN_PREFILL_CONV_TOKEN_TILE - 1) /
                    GDN_PREFILL_CONV_TOKEN_TILE,
                batch_size);
  }

  if (dtype == 0) {
    if (use_width4_tiled) {
      causal_conv1d_full_width4_tiled_kernel<__half,
                                             GDN_PREFILL_CONV_TOKEN_TILE>
          <<<grid, block, 0, custream>>>(
              (const __half *)x, (const __half *)weight,
              (const __half *)conv_state_in, (__half *)output, batch_size,
              conv_dim, seq_len, x_stride_b, x_stride_s, x_stride_c,
              slot_indices);
    } else {
      causal_conv1d_full_kernel<__half><<<grid, block, 0, custream>>>(
          (const __half *)x, (const __half *)weight,
          (const __half *)conv_state_in, (__half *)output, batch_size,
          conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s, x_stride_c,
          slot_indices);
    }
    dim3 grid2((conv_dim + state_block.x - 1) / state_block.x, batch_size);
    save_conv_state_kernel<__half><<<grid2, state_block, 0, custream>>>(
        (const __half *)x, (const __half *)conv_state_in,
        (__half *)conv_state_out, batch_size, conv_dim, seq_len, kernel_size,
        x_stride_b, x_stride_s, x_stride_c, slot_indices);
  } else {
    if (use_width4_tiled) {
      causal_conv1d_full_width4_tiled_kernel<
          __nv_bfloat16, GDN_PREFILL_CONV_TOKEN_TILE>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
              (const __nv_bfloat16 *)conv_state_in,
              (__nv_bfloat16 *)output, batch_size, conv_dim, seq_len,
              x_stride_b, x_stride_s, x_stride_c, slot_indices);
    } else {
      causal_conv1d_full_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
          (const __nv_bfloat16 *)conv_state_in, (__nv_bfloat16 *)output,
          batch_size, conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s,
          x_stride_c, slot_indices);
    }
    dim3 grid2((conv_dim + state_block.x - 1) / state_block.x, batch_size);
    save_conv_state_kernel<__nv_bfloat16>
        <<<grid2, state_block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)conv_state_in,
        (__nv_bfloat16 *)conv_state_out, batch_size, conv_dim, seq_len,
        kernel_size, x_stride_b, x_stride_s, x_stride_c, slot_indices);
  }
}


template <typename T, typename OutT>
static void launch_gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *q_out, void *k_out, void *v_out, float *g_out,
    float *beta_out, int batch_size, int seq_len, int token_stride,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(batch_size * seq_len * num_v_heads);
  gdn_prepare_recurrence_kernel<T, OutT><<<grid, block, 0, stream>>>(
      (const T *)mixed_qkv, (const T *)b, (const T *)a, a_log, dt_bias,
      (OutT *)q_out, (OutT *)k_out, (OutT *)v_out, g_out, beta_out, batch_size,
      seq_len, token_stride, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
      tiled_v_heads);
}

extern "C" void gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *q_out, void *k_out, void *v_out, float *g_out,
    float *beta_out, int batch_size, int seq_len, int token_stride,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int dtype, int out_bf16, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0 && out_bf16) {
    launch_gdn_prepare_recurrence<__half, __nv_bfloat16>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else if (dtype == 0) {
    launch_gdn_prepare_recurrence<__half, float>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else if (out_bf16) {
    launch_gdn_prepare_recurrence<__nv_bfloat16, __nv_bfloat16>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else {
    launch_gdn_prepare_recurrence<__nv_bfloat16, float>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  }
}



template <typename T, typename StateT>
void launch_gdn_decode_recurrence(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, StateT *state, T *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, cudaStream_t stream) {
  if (kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_4 ||
      kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_32) {
    if (head_k_dim == GDN_DECODE_VALUE_MAJOR_K &&
        head_v_dim == GDN_DECODE_VALUE_MAJOR_V) {
      const int bh = batch_size * num_v_heads;
      if (kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_4) {
        gdn_decode_recurrence_kernel_value_major<T, StateT, 4>
            <<<bh * (GDN_DECODE_VALUE_MAJOR_V / 4), 32, 0, stream>>>(
                mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
                num_k_heads, num_v_heads, tiled_v_heads, b_batch_stride,
                b_head_stride, a_batch_stride, a_head_stride, slot_indices);
      } else {
        gdn_decode_recurrence_kernel_value_major<T, StateT, 32>
            <<<bh * (GDN_DECODE_VALUE_MAJOR_V / 32), 32, 0, stream>>>(
                mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
                num_k_heads, num_v_heads, tiled_v_heads, b_batch_stride,
                b_head_stride, a_batch_stride, a_head_stride, slot_indices);
      }
    }
    return;
  }
  constexpr int BV = GDN_DECODE_VALUE_TILE;
  dim3 grid((head_v_dim + BV - 1) / BV, batch_size * num_v_heads);
  dim3 block(BV);

  if (kernel_kind == GDN_DECODE_KERNEL_PIPELINED &&
      head_k_dim == GDN_DECODE_PIPELINED_K &&
      head_v_dim % GDN_DECODE_PIPELINED_V == 0) {
    dim3 pipelined_grid(batch_size * num_v_heads);
    dim3 pipelined_block(GDN_DECODE_PIPELINED_THREADS);
    gdn_decode_recurrence_kernel_pipelined<T, StateT>
        <<<pipelined_grid, pipelined_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (kernel_kind == GDN_DECODE_KERNEL_COOPERATIVE &&
             head_k_dim == GDN_DECODE_COOPERATIVE_K &&
             head_v_dim % GDN_DECODE_COOPERATIVE_V == 0) {
    constexpr int COOPERATIVE_BV = GDN_DECODE_COOPERATIVE_V;
    dim3 cooperative_grid(head_v_dim / COOPERATIVE_BV,
                          batch_size * num_v_heads);
    dim3 cooperative_block(GDN_DECODE_COOPERATIVE_THREADS);
    gdn_decode_recurrence_kernel_cooperative<T, StateT>
        <<<cooperative_grid, cooperative_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (head_k_dim == 128) {
    gdn_decode_recurrence_kernel<T, StateT, 128, BV>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (head_k_dim == 64) {
    gdn_decode_recurrence_kernel<T, StateT, 64, BV>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else {
    constexpr int MAX_K = 256;
    size_t smem = (2 * BV + 2 * head_k_dim) * sizeof(float);
    gdn_decode_recurrence_kernel_fallback<T, StateT, BV, MAX_K>
        <<<grid, block, smem, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  }
}

template <typename T>
void dispatch_gdn_decode_recurrence(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, void *state, T *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, int state_dtype,
    cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (__half *)state, output, batch_size,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
        b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
        slot_indices, kernel_kind, stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (__nv_bfloat16 *)state, output,
        batch_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        tiled_v_heads, b_batch_stride, b_head_stride, a_batch_stride,
        a_head_stride, slot_indices, kernel_kind, stream);
  } else {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (float *)state, output, batch_size,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
        b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
        slot_indices, kernel_kind, stream);
  }
}

extern "C" void gdn_decode_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *state, void *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, int dtype, int state_dtype,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_decode_recurrence(
        (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
        a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride,
        b_head_stride, a_batch_stride, a_head_stride, slot_indices, kernel_kind,
        state_dtype, custream);
  } else {
    dispatch_gdn_decode_recurrence(
        (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a, a_log, dt_bias, state,
        (__nv_bfloat16 *)output, batch_size, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride, b_head_stride,
        a_batch_stride, a_head_stride, slot_indices, kernel_kind, state_dtype,
        custream);
  }
}

#include "gdn_norm_device.cuh"

template <typename T>
__host__ __forceinline__ bool gdn_rmsnorm_vec8_aligned(const void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(gdn_rmsnorm_vec8<T>) == 0;
}

__host__ __forceinline__ bool gdn_rmsnorm_dense_rows(
    int rows, int outer_dim_1, int outer_dim_2, int64_t stride_0,
    int64_t stride_1, int64_t stride_2) {
  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_dim_0 = rows / outer_plane;
  return (outer_dim_2 <= 1 || stride_2 == GDN_RMSNORM_FAST_HIDDEN) &&
         (outer_dim_1 <= 1 ||
          stride_1 == (int64_t)outer_dim_2 * GDN_RMSNORM_FAST_HIDDEN) &&
         (outer_dim_0 <= 1 ||
          stride_0 == (int64_t)outer_plane * GDN_RMSNORM_FAST_HIDDEN);
}

extern "C" void gdn_rmsnorm_gated(const void *x, const void *gate,
                                  const void *weight, void *output, int rows,
                                  int hidden_dim, int outer_dim_1,
                                  int outer_dim_2, int64_t x_stride_0,
                                  int64_t x_stride_1, int64_t x_stride_2,
                                  int64_t x_stride_3, int64_t gate_stride_0,
                                  int64_t gate_stride_1, int64_t gate_stride_2,
                                  int64_t gate_stride_3, float eps,
                                  int sigmoid_gate, int dtype,
                                  int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const bool use_warp_kernel =
      hidden_dim == GDN_RMSNORM_FAST_HIDDEN && x_stride_3 == 1 &&
      gate_stride_3 == 1;
  if (use_warp_kernel) {
    const bool use_rows4_kernel =
        rows >= GDN_RMSNORM_TILED_MIN_ROWS &&
        gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, x_stride_0,
                               x_stride_1, x_stride_2) &&
        gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, gate_stride_0,
                               gate_stride_1, gate_stride_2);
    if (dtype == 0 && use_rows4_kernel &&
        gdn_rmsnorm_vec8_aligned<__half>(x) &&
        gdn_rmsnorm_vec8_aligned<__half>(gate) &&
        gdn_rmsnorm_vec8_aligned<__half>(weight) &&
        gdn_rmsnorm_vec8_aligned<__half>(output)) {
      const dim3 rows4_block(32);
      const dim3 rows4_grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                            GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
      gdn_rmsnorm_gated_rows4_kernel<<<rows4_grid, rows4_block, 0, custream>>>(
          (const __half *)x, (const __half *)gate, (const __half *)weight,
          (__half *)output, rows, eps, sigmoid_gate);
      return;
    }
    if (dtype != 0 && use_rows4_kernel &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(x) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(gate) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(weight) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(output)) {
      const dim3 rows4_block(32);
      const dim3 rows4_grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                            GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
      gdn_rmsnorm_gated_rows4_kernel<<<rows4_grid, rows4_block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows, eps,
          sigmoid_gate);
      return;
    }

    const dim3 warp_block(32, GDN_RMSNORM_ROWS_PER_BLOCK);
    const dim3 warp_grid((rows + GDN_RMSNORM_ROWS_PER_BLOCK - 1) /
                         GDN_RMSNORM_ROWS_PER_BLOCK);
    if (dtype == 0) {
      gdn_rmsnorm_gated_warp_kernel<__half, GDN_RMSNORM_FAST_HIDDEN,
                                     GDN_RMSNORM_ROWS_PER_BLOCK>
          <<<warp_grid, warp_block, 0, custream>>>(
              (const __half *)x, (const __half *)gate,
              (const __half *)weight, (__half *)output, rows, outer_dim_1,
              outer_dim_2, x_stride_0, x_stride_1, x_stride_2, gate_stride_0,
              gate_stride_1, gate_stride_2, eps, sigmoid_gate);
    } else {
      gdn_rmsnorm_gated_warp_kernel<
          __nv_bfloat16, GDN_RMSNORM_FAST_HIDDEN,
          GDN_RMSNORM_ROWS_PER_BLOCK>
          <<<warp_grid, warp_block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
              (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows,
              outer_dim_1, outer_dim_2, x_stride_0, x_stride_1, x_stride_2,
              gate_stride_0, gate_stride_1, gate_stride_2, eps, sigmoid_gate);
    }
    return;
  }

  dim3 block(128);
  dim3 grid(rows);

  if (dtype == 0) {
    gdn_rmsnorm_gated_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)gate, (const __half *)weight,
        (__half *)output, rows, hidden_dim, outer_dim_1, outer_dim_2,
        x_stride_0, x_stride_1, x_stride_2, x_stride_3, gate_stride_0,
        gate_stride_1, gate_stride_2, gate_stride_3, eps, sigmoid_gate);
  } else {
    gdn_rmsnorm_gated_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows,
        hidden_dim, outer_dim_1, outer_dim_2, x_stride_0, x_stride_1,
        x_stride_2, x_stride_3, gate_stride_0, gate_stride_1, gate_stride_2,
        gate_stride_3, eps, sigmoid_gate);
  }
}

extern "C" void gdn_rmsnorm_gated_quantized_bf16(
    const void *x, const void *gate, const void *weight, void *output,
    float *scales, int rows, int groups, int scale_stride_m,
    int scale_layout, int outer_dim_1, int outer_dim_2, int64_t x_stride_0,
    int64_t x_stride_1, int64_t x_stride_2, int64_t x_stride_3,
    int64_t gate_stride_0, int64_t gate_stride_1, int64_t gate_stride_2,
    int64_t gate_stride_3, float eps, int sigmoid_gate, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const bool use_rows4_kernel =
      rows >= GDN_RMSNORM_TILED_MIN_ROWS &&
      x_stride_3 == 1 && gate_stride_3 == 1 &&
      gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, x_stride_0,
                             x_stride_1, x_stride_2) &&
      gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, gate_stride_0,
                             gate_stride_1, gate_stride_2) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(x) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(gate) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(weight);
  if (use_rows4_kernel) {
    const dim3 block(32);
    const dim3 grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                    GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
    gdn_rmsnorm_gated_quantized_rows4_kernel<<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)weight, (gdn_fp8_e4m3 *)output, scales, rows,
        groups, scale_stride_m, scale_layout, eps, sigmoid_gate);
    return;
  }

  const dim3 block(32, GDN_RMSNORM_ROWS_PER_BLOCK);
  const dim3 grid((rows + GDN_RMSNORM_ROWS_PER_BLOCK - 1) /
                  GDN_RMSNORM_ROWS_PER_BLOCK);
  gdn_rmsnorm_gated_quantized_warp_kernel<GDN_RMSNORM_ROWS_PER_BLOCK>
      <<<grid, block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)weight, (gdn_fp8_e4m3 *)output, scales,
          rows, groups, scale_stride_m, scale_layout, outer_dim_1,
          outer_dim_2, x_stride_0, x_stride_1, x_stride_2, x_stride_3,
          gate_stride_0, gate_stride_1, gate_stride_2, gate_stride_3, eps,
          sigmoid_gate);
}

// ============================================================================
// Kernel 3: fused_gdn_gating
//
// Fuses: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
// a_log and dt_bias are per-head (broadcast over batch*seq).
//
// b, a: [total]  a_log, dt_bias: [num_heads]
// beta_out, g_out: [total]
// ============================================================================


extern "C" void fused_gdn_gating(const void *b, const void *a,
                                 const float *a_log, const float *dt_bias,
                                 void *beta_out, void *g_out,
                                 int total_elements, int num_heads, int dtype,
                                 int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(256);
  dim3 grid((total_elements + 255) / 256);

  if (dtype == 0) {
    fused_gdn_gating_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)b, (const __half *)a, a_log, dt_bias,
        (__half *)beta_out, (__half *)g_out, total_elements, num_heads);
  } else {
    fused_gdn_gating_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)b, (const __nv_bfloat16 *)a, a_log, dt_bias,
        (__nv_bfloat16 *)beta_out, (__nv_bfloat16 *)g_out, total_elements,
        num_heads);
  }
}

#include "gdn_checkpoint_device.cuh"

template <typename T, typename StateT>
void launch_gdn_speculative_state_commit(
    const T *mixed_qkv, const T *convolved_qkv, const T *b, const T *a,
    const T *initial_conv_state, const StateT *initial_recurrent_state,
    const float *a_log, const float *dt_bias, T *conv_state_pool,
    StateT *recurrent_state_pool, const uint32_t *keep_rows,
    const uint32_t *slot_indices, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int kernel_size,
    int tiled_v_heads, int value_major, cudaStream_t stream) {
  dim3 conv_block(GDN_CHANNEL_BLOCK_SIZE);
  const int conv_dim =
      2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
  dim3 conv_grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) /
                     GDN_CHANNEL_BLOCK_SIZE,
                 batch_size);
  gdn_speculative_conv_state_commit_kernel<T>
      <<<conv_grid, conv_block, 0, stream>>>(
          mixed_qkv, initial_conv_state, conv_state_pool, keep_rows,
          slot_indices, batch_size, seq_len, conv_dim, kernel_size, 0);

  dim3 recurrence_block(32, GDN_SPEC_COMMIT_WARPS);
  dim3 recurrence_grid(
      (head_v_dim + GDN_SPEC_COMMIT_WARPS - 1) / GDN_SPEC_COMMIT_WARPS,
      batch_size * num_v_heads);
  if (value_major) {
    gdn_speculative_recurrent_state_commit_kernel<T, StateT, true>
        <<<recurrence_grid, recurrence_block, 0, stream>>>(
            convolved_qkv, b, a, initial_recurrent_state, a_log, dt_bias,
            recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            tiled_v_heads, 0);
  } else {
    gdn_speculative_recurrent_state_commit_kernel<T, StateT, false>
        <<<recurrence_grid, recurrence_block, 0, stream>>>(
            convolved_qkv, b, a, initial_recurrent_state, a_log, dt_bias,
            recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            tiled_v_heads, 0);
  }
}

template <typename T>
void dispatch_gdn_speculative_state_commit(
    const T *mixed_qkv, const T *convolved_qkv, const T *b, const T *a,
    const T *initial_conv_state, const void *initial_recurrent_state,
    const float *a_log, const float *dt_bias, T *conv_state_pool,
    void *recurrent_state_pool, const uint32_t *keep_rows,
    const uint32_t *slot_indices, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int kernel_size,
    int tiled_v_heads, int value_major, int state_dtype, cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const __half *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (__half *)recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const __nv_bfloat16 *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (__nv_bfloat16 *)recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        stream);
  } else {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const float *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (float *)recurrent_state_pool, keep_rows, slot_indices,
        batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        kernel_size, tiled_v_heads, value_major, stream);
  }
}

extern "C" void gdn_speculative_state_commit(
    const void *mixed_qkv, const void *convolved_qkv, const void *b,
    const void *a, const void *initial_conv_state,
    const void *initial_recurrent_state, const float *a_log,
    const float *dt_bias, void *conv_state_pool, void *recurrent_state_pool,
    const uint32_t *keep_rows, const uint32_t *slot_indices, int batch_size,
    int seq_len, int num_k_heads, int num_v_heads, int head_k_dim,
    int head_v_dim, int kernel_size, int tiled_v_heads, int value_major,
    int dtype, int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_speculative_state_commit(
        (const __half *)mixed_qkv, (const __half *)convolved_qkv,
        (const __half *)b, (const __half *)a,
        (const __half *)initial_conv_state, initial_recurrent_state, a_log,
        dt_bias, (__half *)conv_state_pool, recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        state_dtype, custream);
  } else {
    dispatch_gdn_speculative_state_commit(
        (const __nv_bfloat16 *)mixed_qkv,
        (const __nv_bfloat16 *)convolved_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a,
        (const __nv_bfloat16 *)initial_conv_state, initial_recurrent_state,
        a_log, dt_bias, (__nv_bfloat16 *)conv_state_pool,
        recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, kernel_size,
        tiled_v_heads, value_major, state_dtype, custream);
  }
}

#include "gdn_speculative_device.cuh"


template <typename T>
void launch_gdn_speculative_conv_checkpoints(
    const T *x, const T *weight, T *state_pool, T *output,
    const uint32_t *active_slots, int batch_size, int seq_len, int conv_dim,
    int kernel_size, int checkpoint_lanes, bool write_checkpoints,
    T *pending_conv_input, const uint32_t *pending_keep_rows,
    const uint32_t *pending_epochs, uint32_t *conv_applied_epochs,
    int max_pending_rows, int pending_capacity, int conv_blocks,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    cudaStream_t stream) {
  if (pending_conv_input != nullptr &&
      (write_checkpoints || max_pending_rows < seq_len ||
       max_pending_rows > GDN_SPEC_FUSED_MAX_TOKENS ||
       pending_capacity <= 0 || conv_blocks <= 0 ||
       pending_keep_rows == nullptr || pending_epochs == nullptr ||
       conv_applied_epochs == nullptr)) {
    return;
  }
  dim3 block(GDN_CHANNEL_BLOCK_SIZE);
  dim3 grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) /
                GDN_CHANNEL_BLOCK_SIZE,
            batch_size);
  if (kernel_size == GDN_PACKED_CONV_WIDTH) {
    gdn_speculative_conv_checkpoints_width4_kernel<T>
        <<<grid, block, 0, stream>>>(
            x, weight, state_pool, output, active_slots, batch_size, seq_len,
            conv_dim, checkpoint_lanes, write_checkpoints, pending_conv_input,
            pending_keep_rows, pending_epochs, conv_applied_epochs,
            max_pending_rows, pending_capacity, conv_blocks, x_stride_b,
            x_stride_s, x_stride_c, 0);
  } else {
    gdn_speculative_conv_checkpoints_kernel<T><<<grid, block, 0, stream>>>(
        x, weight, state_pool, output, active_slots, batch_size, seq_len,
        conv_dim, kernel_size, checkpoint_lanes, write_checkpoints,
        pending_conv_input, pending_keep_rows, pending_epochs,
        conv_applied_epochs, max_pending_rows, pending_capacity, conv_blocks,
        x_stride_b, x_stride_s, x_stride_c, 0);
  }
}

extern "C" void gdn_speculative_conv_checkpoints(
    const void *x, const void *weight, void *state_pool, void *output,
    const uint32_t *active_slots, int batch_size, int seq_len, int conv_dim,
    int kernel_size, int checkpoint_lanes, int write_checkpoints,
    void *pending_conv_input, const uint32_t *pending_keep_rows,
    const uint32_t *pending_epochs, uint32_t *conv_applied_epochs,
    int max_pending_rows, int pending_capacity, int conv_blocks,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c, int dtype,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    launch_gdn_speculative_conv_checkpoints(
        (const __half *)x, (const __half *)weight, (__half *)state_pool,
        (__half *)output, active_slots, batch_size, seq_len, conv_dim,
        kernel_size, checkpoint_lanes, write_checkpoints != 0,
        (__half *)pending_conv_input, pending_keep_rows, pending_epochs,
        conv_applied_epochs, max_pending_rows, pending_capacity, conv_blocks,
        x_stride_b, x_stride_s, x_stride_c, custream);
  } else {
    launch_gdn_speculative_conv_checkpoints(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)state_pool, (__nv_bfloat16 *)output, active_slots,
        batch_size, seq_len, conv_dim, kernel_size, checkpoint_lanes,
        write_checkpoints != 0,
        (__nv_bfloat16 *)pending_conv_input, pending_keep_rows,
        pending_epochs, conv_applied_epochs, max_pending_rows,
        pending_capacity, conv_blocks, x_stride_b, x_stride_s, x_stride_c,
        custream);
  }
}


extern "C" void gdn_deferred_recurrence_rmsnorm_gate_value_major_128(
    const void *mixed_qkv, const void *b, const void *a,
    const float *a_log, const float *dt_bias, float *state_pool, void *output,
    void *quantized_output, float *output_scales, int scale_stride_m,
    int scale_layout,
    const uint32_t *active_slots, float *deferred_key,
    float *deferred_delta, float *deferred_decay, uint32_t *deferred_cursor,
    const void *gate, const void *norm_weight, int64_t b_stride_b,
    int64_t b_stride_h, int64_t a_stride_b, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_h, int64_t gate_stride_v,
    int batch_size, int capacity, int num_k_heads, int num_v_heads,
    int tiled_v_heads, float norm_eps, int sigmoid_gate, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 recurrence_grid(batch_size, num_v_heads);
  gdn_deferred_recurrence_rmsnorm_gate_value_major_128_kernel
      <<<recurrence_grid, GDN_SPEC_FUSED_THREADS, 0, custream>>>(
          (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
          (const __nv_bfloat16 *)a, a_log, dt_bias, state_pool,
          (__nv_bfloat16 *)output, (gdn_fp8_e4m3 *)quantized_output,
          output_scales, scale_stride_m, scale_layout, active_slots,
          deferred_key, deferred_delta, deferred_decay, deferred_cursor,
          (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)norm_weight, b_stride_b, b_stride_h,
          a_stride_b, a_stride_h, gate_stride_b, gate_stride_h,
          gate_stride_v, batch_size, capacity, num_k_heads, num_v_heads,
          tiled_v_heads, norm_eps, sigmoid_gate);
  constexpr int FINALIZE_THREADS = 256;
  gdn_deferred_cursor_advance_kernel
      <<<(batch_size + FINALIZE_THREADS - 1) / FINALIZE_THREADS,
         FINALIZE_THREADS, 0, custream>>>(deferred_cursor, active_slots,
                                          batch_size, capacity);
}

extern "C" void gdn_flush_deferred_state_value_major_128(
    float *state_pool, const uint32_t *active_slots,
    const float *deferred_key, const float *deferred_delta,
    const float *deferred_decay, uint32_t *deferred_cursor, int batch_size,
    int capacity, int num_k_heads, int num_v_heads, int tiled_v_heads,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 flush_grid(batch_size, num_v_heads);
  gdn_flush_deferred_state_value_major_128_kernel
      <<<flush_grid, GDN_SPEC_FUSED_THREADS, 0, custream>>>(
          state_pool, active_slots, deferred_key, deferred_delta,
          deferred_decay, deferred_cursor, batch_size, capacity, num_k_heads,
          num_v_heads, tiled_v_heads);
  constexpr int FINALIZE_THREADS = 256;
  gdn_deferred_cursor_clear_kernel
      <<<(batch_size + FINALIZE_THREADS - 1) / FINALIZE_THREADS,
         FINALIZE_THREADS, 0, custream>>>(deferred_cursor, active_slots,
                                          batch_size, capacity);
}


template <typename T, typename StateT>
void launch_gdn_speculative_recurrence_checkpoints(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, StateT *state_pool, T *output,
    gdn_fp8_e4m3 *quantized_output, float *output_scales,
    int scale_stride_m, int scale_layout,
    const uint32_t *active_slots, const T *gate, const T *norm_weight,
    float *transition_key, float *transition_delta, float *transition_decay,
    float *pending_key_banks, const uint32_t *pending_key_bank,
    float *pending_delta, float *pending_decay,
    const uint32_t *pending_keep_rows, const uint32_t *pending_epochs,
    uint32_t *recurrent_applied_epochs,
    int max_pending_rows, int pending_capacity,
    int slot_indexed_transitions,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_s, int64_t gate_stride_h,
    int64_t gate_stride_v, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int value_major, float norm_eps, int sigmoid_gate,
    cudaStream_t stream) {
  const bool batch_transitions = transition_delta != nullptr;
  const bool direct_transitions = slot_indexed_transitions != 0;
  if ((transition_key != nullptr) != batch_transitions ||
      (transition_decay != nullptr) != batch_transitions) {
    return;
  }
  if ((quantized_output != nullptr) != (output_scales != nullptr)) {
    return;
  }
  const bool has_pending =
      pending_key_banks != nullptr || pending_key_bank != nullptr ||
      pending_delta != nullptr || pending_decay != nullptr ||
      pending_keep_rows != nullptr || pending_epochs != nullptr ||
      recurrent_applied_epochs != nullptr;
  if (direct_transitions &&
      (batch_transitions || !has_pending || pending_key_banks == nullptr ||
       pending_key_bank == nullptr || pending_delta == nullptr ||
       pending_decay == nullptr || pending_keep_rows == nullptr ||
       pending_epochs == nullptr || recurrent_applied_epochs == nullptr ||
       max_pending_rows <= 0 ||
       max_pending_rows > GDN_SPEC_FUSED_MAX_TOKENS ||
       pending_capacity <= 0)) {
    return;
  }
  if (!direct_transitions && has_pending) {
    return;
  }
  if (gate != nullptr && norm_weight != nullptr) {
    dim3 fused_grid(batch_size, num_v_heads);
    const int grid_blocks = batch_size * num_v_heads;
    const bool paired_reductions =
        sizeof(StateT) < sizeof(float) ||
        grid_blocks <= GDN_SPEC_FUSED_PAIR_LOW_GRID_MAX ||
        grid_blocks >= GDN_SPEC_FUSED_PAIR_HIGH_GRID_MIN;
#define GDN_LAUNCH_SPEC_RECURRENCE(PAIRED, DIRECT)                           \
  gdn_speculative_recurrence_rmsnorm_gate_value_major_128_kernel<           \
      T, StateT, PAIRED, DIRECT><<<fused_grid, GDN_SPEC_FUSED_THREADS, 0,    \
                                   stream>>>(                               \
      mixed_qkv, b, a, a_log, dt_bias, state_pool, output,                 \
      quantized_output, output_scales, scale_stride_m, scale_layout,        \
      active_slots,                                                        \
      gate, norm_weight, transition_key, transition_delta, transition_decay, \
      pending_key_banks, pending_key_bank, pending_delta, pending_decay,     \
      pending_keep_rows, pending_epochs, recurrent_applied_epochs,           \
      max_pending_rows, pending_capacity, b_stride_b, b_stride_s,            \
      b_stride_h, a_stride_b, a_stride_s, a_stride_h, gate_stride_b,         \
      gate_stride_s, gate_stride_h, gate_stride_v, batch_size, seq_len,      \
      num_k_heads, num_v_heads, checkpoint_lanes, tiled_v_heads, norm_eps, \
      sigmoid_gate, 0)
    if (direct_transitions) {
      if (paired_reductions) {
        GDN_LAUNCH_SPEC_RECURRENCE(true, true);
      } else {
        GDN_LAUNCH_SPEC_RECURRENCE(false, true);
      }
    } else if (paired_reductions) {
      GDN_LAUNCH_SPEC_RECURRENCE(true, false);
    } else {
      GDN_LAUNCH_SPEC_RECURRENCE(false, false);
    }
#undef GDN_LAUNCH_SPEC_RECURRENCE
    return;
  }

  dim3 grid((head_v_dim + GDN_SPEC_CHECKPOINT_VALUE_TILE - 1) /
                GDN_SPEC_CHECKPOINT_VALUE_TILE,
            batch_size * num_v_heads);
  if (value_major && head_k_dim == GDN_DECODE_VALUE_MAJOR_K &&
      head_v_dim == GDN_DECODE_VALUE_MAJOR_V) {
    dim3 block(32, GDN_SPEC_CHECKPOINT_WARPS);
    gdn_speculative_recurrence_checkpoints_value_major_128_kernel<T, StateT>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state_pool, output, active_slots,
            b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
            a_stride_h,
            batch_size, seq_len, num_k_heads, num_v_heads, checkpoint_lanes,
            tiled_v_heads, 0);
    return;
  }

  dim3 fallback_block(GDN_SPEC_CHECKPOINT_VALUE_TILE);
  if (value_major) {
    gdn_speculative_recurrence_checkpoints_fallback_kernel<T, StateT, true>
        <<<grid, fallback_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state_pool, output, active_slots,
            b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
            a_stride_h,
            batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim,
            head_v_dim, checkpoint_lanes, tiled_v_heads, 0);
  } else {
    gdn_speculative_recurrence_checkpoints_fallback_kernel<T, StateT, false>
        <<<grid, fallback_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state_pool, output, active_slots,
            b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
            a_stride_h,
            batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim,
            head_v_dim, checkpoint_lanes, tiled_v_heads, 0);
  }
}

template <typename T>
void dispatch_gdn_speculative_recurrence_checkpoints(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, void *state_pool, T *output,
    gdn_fp8_e4m3 *quantized_output, float *output_scales,
    int scale_stride_m, int scale_layout,
    const uint32_t *active_slots, const T *gate, const T *norm_weight,
    float *transition_key, float *transition_delta, float *transition_decay,
    float *pending_key_banks, const uint32_t *pending_key_bank,
    float *pending_delta, float *pending_decay,
    const uint32_t *pending_keep_rows, const uint32_t *pending_epochs,
    uint32_t *recurrent_applied_epochs,
    int max_pending_rows, int pending_capacity,
    int slot_indexed_transitions,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_s, int64_t gate_stride_h,
    int64_t gate_stride_v, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int value_major, float norm_eps, int sigmoid_gate, int state_dtype,
    cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_speculative_recurrence_checkpoints(
        mixed_qkv, b, a, a_log, dt_bias, (__half *)state_pool, output,
        quantized_output, output_scales, scale_stride_m, scale_layout,
        active_slots, gate, norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, sigmoid_gate, stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_speculative_recurrence_checkpoints(
        mixed_qkv, b, a, a_log, dt_bias, (__nv_bfloat16 *)state_pool, output,
        quantized_output, output_scales, scale_stride_m, scale_layout,
        active_slots, gate, norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, sigmoid_gate, stream);
  } else {
    launch_gdn_speculative_recurrence_checkpoints(
        mixed_qkv, b, a, a_log, dt_bias, (float *)state_pool, output,
        quantized_output, output_scales, scale_stride_m, scale_layout,
        active_slots, gate, norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, sigmoid_gate, stream);
  }
}

extern "C" void gdn_speculative_recurrence_checkpoints(
    const void *mixed_qkv, const void *b, const void *a,
    const float *a_log, const float *dt_bias, void *state_pool, void *output,
    void *quantized_output, float *output_scales, int scale_stride_m,
    int scale_layout,
    const uint32_t *active_slots, const void *gate, const void *norm_weight,
    float *transition_key, float *transition_delta, float *transition_decay,
    float *pending_key_banks, const uint32_t *pending_key_bank,
    float *pending_delta, float *pending_decay,
    const uint32_t *pending_keep_rows, const uint32_t *pending_epochs,
    uint32_t *recurrent_applied_epochs,
    int max_pending_rows, int pending_capacity,
    int slot_indexed_transitions,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_s, int64_t gate_stride_h,
    int64_t gate_stride_v, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int value_major, float norm_eps, int sigmoid_gate, int dtype,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_speculative_recurrence_checkpoints(
        (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
        a_log, dt_bias, state_pool, (__half *)output,
        (gdn_fp8_e4m3 *)quantized_output, output_scales, scale_stride_m,
        scale_layout, active_slots,
        (const __half *)gate, (const __half *)norm_weight, transition_key,
        transition_delta, transition_decay, pending_key_banks,
        pending_key_bank, pending_delta, pending_decay, pending_keep_rows,
        pending_epochs, recurrent_applied_epochs, max_pending_rows,
        pending_capacity, slot_indexed_transitions, b_stride_b, b_stride_s,
        b_stride_h, a_stride_b, a_stride_s, a_stride_h, gate_stride_b,
        gate_stride_s, gate_stride_h, gate_stride_v, batch_size, seq_len,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, checkpoint_lanes,
        tiled_v_heads, value_major, norm_eps, sigmoid_gate, state_dtype, custream);
  } else {
    dispatch_gdn_speculative_recurrence_checkpoints(
        (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a, a_log, dt_bias, state_pool,
        (__nv_bfloat16 *)output, (gdn_fp8_e4m3 *)quantized_output,
        output_scales, scale_stride_m, scale_layout, active_slots,
        (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, sigmoid_gate, state_dtype, custream);
  }
}

#include "gdn_transition_device.cuh"
static_assert(GDN_TX_MAX_ROWS == GDN_SPEC_FUSED_MAX_TOKENS);
static_assert(GDN_TX_MAX_CONV_WIDTH == GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH);
static_assert(GDN_TX_KEY_DIM == GDN_DECODE_VALUE_MAJOR_K && GDN_TX_VALUE_DIM == GDN_DECODE_VALUE_MAJOR_V);
static_assert(GDN_TX_PAD_SLOT == GDN_SPEC_CHECKPOINT_PAD_SLOT && GDN_TX_BANK_MASK == GDN_PENDING_KEY_BANK_MASK);

template <typename T>
void dispatch_gdn_speculative_transition_commit_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *active_slots, int layer_count, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int value_major,
    int state_dtype, cudaStream_t stream) {
  const int conv_blocks =
      (conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE;
  dim3 grid(conv_blocks + num_v_heads, batch_size, layer_count);
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    gdn_speculative_transition_commit_batched_kernel<T, __half>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, keep_rows, active_slots, layer_count, batch_size,
            seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, value_major, 0);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    gdn_speculative_transition_commit_batched_kernel<T, __nv_bfloat16>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, keep_rows, active_slots, layer_count, batch_size,
            seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, value_major, 0);
  } else {
    gdn_speculative_transition_commit_batched_kernel<T, float>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, keep_rows, active_slots, layer_count, batch_size,
            seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, value_major, 0);
  }
}

extern "C" void gdn_speculative_transition_commit_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *active_slots, int layer_count, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int value_major,
    int activation_dtype, int state_dtype, int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || seq_len <= 0 ||
      seq_len > GDN_SPEC_FUSED_MAX_TOKENS ||
      head_k_dim != GDN_DECODE_VALUE_MAJOR_K ||
      head_v_dim != GDN_DECODE_VALUE_MAJOR_V) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  if (activation_dtype == 0) {
    dispatch_gdn_speculative_transition_commit_batched<__half>(
        pointer_table, keep_rows, active_slots, layer_count, batch_size,
        seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, value_major, state_dtype, custream);
  } else {
    dispatch_gdn_speculative_transition_commit_batched<__nv_bfloat16>(
        pointer_table, keep_rows, active_slots, layer_count, batch_size,
        seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, value_major, state_dtype, custream);
  }
}


template <typename T>
void dispatch_gdn_speculative_transition_stage_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *destination_slots, int layer_count, int batch_size,
    int seq_len, int max_rows,
    int destination_capacity, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_dim, cudaStream_t stream) {
  const size_t row_elements =
      (size_t)conv_dim + (size_t)num_k_heads * head_k_dim +
      (size_t)num_v_heads * head_v_dim + num_v_heads;
  const size_t required_blocks =
      ((size_t)seq_len * row_elements + GDN_CHANNEL_BLOCK_SIZE - 1) /
      GDN_CHANNEL_BLOCK_SIZE;
  const unsigned int copy_blocks = (unsigned int)(
      required_blocks < GDN_TRANSITION_STAGE_COPY_BLOCKS
          ? required_blocks
          : GDN_TRANSITION_STAGE_COPY_BLOCKS);
  dim3 grid(copy_blocks, batch_size, layer_count);
  gdn_speculative_transition_stage_batched_kernel<T>
      <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
          pointer_table, keep_rows, destination_slots, layer_count, batch_size,
          seq_len, max_rows, destination_capacity, num_k_heads,
          num_v_heads, head_k_dim, head_v_dim, conv_dim);
}

extern "C" void gdn_speculative_transition_stage_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *destination_slots, int layer_count, int batch_size,
    int seq_len, int max_rows,
    int destination_capacity, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_dim, int activation_dtype,
    int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || seq_len <= 0 ||
      seq_len > GDN_SPEC_FUSED_MAX_TOKENS || max_rows < seq_len ||
      max_rows > GDN_SPEC_FUSED_MAX_TOKENS || destination_capacity <= 0 ||
      num_k_heads <= 0 || num_v_heads <= 0 || head_k_dim <= 0 ||
      head_v_dim <= 0 || conv_dim <= 0) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  if (activation_dtype == 0) {
    dispatch_gdn_speculative_transition_stage_batched<__half>(
        pointer_table, keep_rows, destination_slots, layer_count, batch_size,
        seq_len, max_rows, destination_capacity, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, conv_dim, custream);
  } else {
    dispatch_gdn_speculative_transition_stage_batched<__nv_bfloat16>(
        pointer_table, keep_rows, destination_slots, layer_count, batch_size,
        seq_len, max_rows, destination_capacity, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, conv_dim, custream);
  }
}


extern "C" void gdn_pending_transition_publish_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *destination_slots, int layer_count, int batch_size,
    int max_rows, int destination_capacity, int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || max_rows <= 0 ||
      max_rows > GDN_SPEC_FUSED_MAX_TOKENS || destination_capacity <= 0) {
    return;
  }
  dim3 grid(batch_size, layer_count);
  gdn_pending_transition_publish_batched_kernel<<<
      grid, 1, 0, (cudaStream_t)stream>>>(
      pointer_table, keep_rows, destination_slots, layer_count, batch_size,
      max_rows, destination_capacity);
}


template <typename T>
void dispatch_gdn_pending_transition_apply_batched(
    const uint64_t *pointer_table, const uint32_t *active_slots,
    int layer_count, int batch_size, int max_rows, int capacity,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int state_dtype,
    cudaStream_t stream) {
  const int conv_blocks =
      (conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE;
  dim3 grid(conv_blocks + num_v_heads, batch_size, layer_count);
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    gdn_pending_transition_apply_batched_kernel<T, __half>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, active_slots, layer_count, batch_size, max_rows,
            capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, conv_blocks);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    gdn_pending_transition_apply_batched_kernel<T, __nv_bfloat16>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, active_slots, layer_count, batch_size, max_rows,
            capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, conv_blocks);
  } else {
    gdn_pending_transition_apply_batched_kernel<T, float>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, active_slots, layer_count, batch_size, max_rows,
            capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, conv_blocks);
  }
}

extern "C" void gdn_pending_transition_apply_batched(
    const uint64_t *pointer_table, const uint32_t *active_slots,
    int layer_count, int batch_size, int max_rows, int capacity,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int activation_dtype,
    int state_dtype, int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || max_rows <= 0 ||
      max_rows > GDN_SPEC_FUSED_MAX_TOKENS || capacity <= 0 ||
      num_k_heads <= 0 || num_v_heads <= 0 ||
      num_v_heads % num_k_heads != 0 ||
      head_k_dim != GDN_DECODE_VALUE_MAJOR_K ||
      head_v_dim != GDN_DECODE_VALUE_MAJOR_V || conv_dim <= 0 ||
      conv_width <= 0 || conv_width > GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  if (activation_dtype == 0) {
    dispatch_gdn_pending_transition_apply_batched<__half>(
        pointer_table, active_slots, layer_count, batch_size, max_rows,
        capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, state_dtype, custream);
  } else {
    dispatch_gdn_pending_transition_apply_batched<__nv_bfloat16>(
        pointer_table, active_slots, layer_count, batch_size, max_rows,
        capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, state_dtype, custream);
  }
}
