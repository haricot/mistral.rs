/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
// DKR-only adaptation of the vendored BatchDecodeWithPagedKVCacheDevice.
// Original shared helper functions and pipeline are retained. See S9B4d5.
// Never included by the static CUDA producer.
namespace flashinfer {
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void DkrSafeBatchDecodeDevice(const Params& params, uint8_t smem[],
                                                             const uint32_t bx = blockIdx.x,
                                                             const uint32_t by = blockIdx.y,
                                                             const uint32_t tx = threadIdx.x,
                                                             const uint32_t ty = threadIdx.y,
                                                             const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  constexpr size_t stage_bytes = 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                 head_dim * sizeof(DTypeKV);
  constexpr size_t state_bytes = bdz > 1 ? bdy * bdz * head_dim * sizeof(float) : 0;
  // State vectors and m/d are live together after the K/V pipeline drains.
  float* smem_md = reinterpret_cast<float*>(smem + (stage_bytes > state_bytes ? stage_bytes : state_bytes));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]) +
            chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  if constexpr (bdz > 1) {
    float* states = reinterpret_cast<float*>(smem);
    st.o.store(states + (tz*bdy+ty)*head_dim + tx*vec_size);
    if (tx == 0) {
      smem_md[(tz*bdy+ty)*2] = st.m;
      smem_md[(tz*bdy+ty)*2+1] = st.d;
    }
    block.sync();
    st.init();
    #pragma unroll
    for (uint32_t j=0;j<bdz;++j) {
      const float mz=smem_md[(j*bdy+ty)*2], dz=smem_md[(j*bdy+ty)*2+1];
      if (mz != -math::inf) {
        vec_t<float,vec_size> oz;
        oz.load(states + (j*bdy+ty)*head_dim + tx*vec_size);
        st.merge(oz,mz,dz);
      }
    }
  }
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    st.o[i] = variant.OutputTransform(params, st.o[i], bx, /*qo_idx=*/0, qo_head_idx, st.m, st.d,
                                      /*scale=*/1.0f);
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr && tx == 0) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

} // namespace flashinfer
