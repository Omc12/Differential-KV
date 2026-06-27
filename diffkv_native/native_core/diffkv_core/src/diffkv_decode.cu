// diffkv_core/src/diffkv_decode.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include "runtime/diffkv_attention.hpp"

namespace diffkv {

#if defined(GGML_USE_CUDA) || defined(USE_CUDA)

// ── Warp-level and block-level reductions in CUDA ───────────────────────────
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float blockReduceMax(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared_mem[lane] : -1e30f;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ inline float blockReduceSum(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared_mem[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

struct SoftmaxState {
    float m;
    float d;
};

__device__ inline SoftmaxState merge_softmax_states(SoftmaxState lhs, SoftmaxState rhs) {
    float m_new = fmaxf(lhs.m, rhs.m);
    float d_new = lhs.d * expf(lhs.m - m_new) + rhs.d * expf(rhs.m - m_new);
    return { m_new, d_new };
}

// ── CUDA Attention Kernel ──────────────────────────────────────────────────
__global__ void decode_attention_cuda_kernel(
    const float*    Q,
    const int8_t*   U_pool,
    const float*    U_scale_pool,
    const half*     VK_pool,
    const half*     VV_pool,
    const half*     anchors_K,
    const half*     anchors_V,
    const int32_t*  seq_lens,
    const int32_t*  slot_indices,
    float*          out_buf,
    float*          lse_buf,
    int32_t         n_q_heads,
    int32_t         n_kv_heads,
    int32_t         rank,
    int32_t         S_max,
    int32_t         K,
    int32_t         D,
    float           scale,
    const half*     scales,
    int32_t         has_rope,
    float           rope_freq_base,
    const int32_t*  anchor_positions,
    const float*    dense_K,
    const float*    dense_V,
    const int32_t*  dense_positions,
    int32_t         T_dense,
    int32_t         approximate_attn,
    float*          split_out,
    float*          split_m,
    float*          split_d,
    int32_t         S_split
) {
    int tg_idx_x = blockIdx.x; // query head index
    int tg_idx_y = blockIdx.y; // split index
    int tid = threadIdx.x;
    int t_per_tg = blockDim.x;

    if (tg_idx_x >= n_q_heads) return;

    const int g = n_q_heads / n_kv_heads;
    const int kv_head = tg_idx_x / g;
    const int half_d = D / 2;

    // Shared Memory declarations
    extern __shared__ char shared_raw[];
    float* q_shared = (float*)shared_raw; // size D floats
    float* red_mem = (float*)(shared_raw + D * sizeof(float)); // size 64 floats
    float* ak_rot_shared = (float*)(shared_raw + (D + 64) * sizeof(float)); // size D floats

    // Cache query
    for (int d = tid; d < D; d += t_per_tg) {
        q_shared[d] = Q[tg_idx_x * D + d];
    }
    __syncthreads();

    SoftmaxState sm_state = { -1e30f, 0.0f };

    const int blocks_per_split = (K + S_split - 1) / S_split;
    const int k_start = tg_idx_y * blocks_per_split;
    const int k_end   = min(k_start + blocks_per_split, K);

    // Pass 1: Sparse blocks
    for (int k = k_start; k < k_end; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];
        int anchor_pos = anchor_positions[slot_id];

        // Rotate anchor key
        for (int d = tid; d < D; d += t_per_tg) {
            float raw_ak = __half2float(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
            if (has_rope) {
                int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                int idx = (d < half_d) ? d : (d - half_d);
                float theta = 1.0f / powf(rope_freq_base, (2.0f * idx) / D);
                float angle = anchor_pos * theta;
                float c = cosf(angle), s = sinf(angle);
                float raw_p = __half2float(anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner]);
                float rot_c = (d < half_d) ? -raw_p : raw_p;
                ak_rot_shared[d] = raw_ak * c + rot_c * s;
            } else {
                ak_rot_shared[d] = raw_ak;
            }
        }
        __syncthreads();

        float thread_anc = 0.0f;
        for (int d = tid; d < D; d += t_per_tg) {
            thread_anc += q_shared[d] * ak_rot_shared[d];
        }
        float score_anc = blockReduceMax(thread_anc, red_mem);
        sm_state = merge_softmax_states(sm_state, { score_anc * scale, 1.0f });
    }

    // Pass 1b: Dense window tokens
    if (T_dense > 0 && tg_idx_y == 0) {
        for (int t = tid; t < T_dense; t += t_per_tg) {
            int pos = dense_positions[t];
            int base_k = t * n_kv_heads * D + kv_head * D;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                float raw_k = dense_K[base_k + d];
                float k_rot = raw_k;
                if (has_rope) {
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    int idx = (d < half_d) ? d : (d - half_d);
                    float theta = 1.0f / powf(rope_freq_base, (2.0f * idx) / D);
                    float angle = pos * theta;
                    float c = cosf(angle), s = sinf(angle);
                    float raw_p = dense_K[base_k + partner];
                    float rot_c = (d < half_d) ? -raw_p : raw_p;
                    k_rot = raw_k * c + rot_c * s;
                }
                score += q_shared[d] * k_rot;
            }
            sm_state = merge_softmax_states(sm_state, { score * scale, 1.0f });
        }
    }

    float global_m = blockReduceMax(sm_state.m, red_mem);
    float adjusted_d = sm_state.d * expf(sm_state.m - global_m);
    float global_d = blockReduceSum(adjusted_d, red_mem);

    if (S_split == 1 && tid == 0) {
        lse_buf[tg_idx_x] = global_m + logf(fmaxf(global_d, 1e-9f));
    }
    __syncthreads();

    // Pass 2: Accumulate outputs
    float thread_val[128] = { 0.0f };
    float norm_factor = (S_split == 1) ? fmaxf(global_d, 1e-9f) : 1.0f;

    for (int k = k_start; k < k_end; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];
        float block_scale = __half2float(scales[slot_id]);
        int anchor_pos = anchor_positions[slot_id];

        // Recompute anchor score
        float score_anc = 0.0f;
        for (int d = 0; d < D; ++d) {
            float raw_ak = __half2float(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
            float ak_rot = raw_ak;
            if (has_rope) {
                int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                int idx = (d < half_d) ? d : (d - half_d);
                float theta = 1.0f / powf(rope_freq_base, (2.0f * idx) / D);
                float angle = anchor_pos * theta;
                float c = cosf(angle), s = sinf(angle);
                float raw_p = __half2float(anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner]);
                float rot_c = (d < half_d) ? -raw_p : raw_p;
                ak_rot = raw_ak * c + rot_c * s;
            }
            score_anc += q_shared[d] * ak_rot;
        }

        float w_anc = expf(score_anc * scale - global_m) / norm_factor;
        float v_anc = __half2float(anchors_V[slot_id * n_kv_heads * D + kv_head * D + tid]);
        thread_val[tid] += w_anc * v_anc;
    }

    // Pass 2b: Dense window tokens
    if (T_dense > 0 && tg_idx_y == 0) {
        for (int t = tid; t < T_dense; t += t_per_tg) {
            int pos = dense_positions[t];
            int base_k = t * n_kv_heads * D + kv_head * D;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                float raw_k = dense_K[base_k + d];
                float k_rot = raw_k;
                if (has_rope) {
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    int idx = (d < half_d) ? d : (d - half_d);
                    float theta = 1.0f / powf(rope_freq_base, (2.0f * idx) / D);
                    float angle = pos * theta;
                    float c = cosf(angle), s = sinf(angle);
                    float raw_p = dense_K[base_k + partner];
                    float rot_c = (d < half_d) ? -raw_p : raw_p;
                    k_rot = raw_k * c + rot_c * s;
                }
                score += q_shared[d] * k_rot;
            }
            float w_dense = expf(score * scale - global_m) / norm_factor;
            float v_dense = dense_V[t * n_kv_heads * D + kv_head * D + tid];
            thread_val[tid] += w_dense * v_dense;
        }
    }

    if (S_split == 1) {
        out_buf[tg_idx_x * D + tid] = thread_val[tid];
    } else {
        split_out[tg_idx_x * S_split * D + tg_idx_y * D + tid] = thread_val[tid];
        if (tid == 0) {
            split_m[tg_idx_x * S_split + tg_idx_y] = global_m;
            split_d[tg_idx_x * S_split + tg_idx_y] = global_d;
        }
    }
}

// ── CUDA Merge Kernel ──────────────────────────────────────────────────────
__global__ void merge_split_k_cuda_kernel(
    const float* split_out,
    const float* split_m,
    const float* split_d,
    float*       out_buf,
    float*       lse_buf,
    int32_t      S_split,
    int32_t      D
) {
    int tg_idx = blockIdx.x; // query head index
    int tid = threadIdx.x;

    float global_m = -1e30f;
    for (int s = 0; s < S_split; ++s) {
        float m_val = split_m[tg_idx * S_split + s];
        global_m = fmaxf(global_m, m_val);
    }

    float global_d = 0.0f;
    for (int s = 0; s < S_split; ++s) {
        float m_val = split_m[tg_idx * S_split + s];
        float d_val = split_d[tg_idx * S_split + s];
        global_d += d_val * expf(m_val - global_m);
    }

    float val_accum = 0.0f;
    for (int s = 0; s < S_split; ++s) {
        float m_val = split_m[tg_idx * S_split + s];
        float local_out = split_out[tg_idx * S_split * D + s * D + tid];
        val_accum += local_out * expf(m_val - global_m);
    }
    out_buf[tg_idx * D + tid] = val_accum / fmaxf(global_d, 1e-9f);

    if (tid == 0) {
        lse_buf[tg_idx] = global_m + logf(fmaxf(global_d, 1e-9f));
    }
}

// ── C++ Host Interface ──────────────────────────────────────────────────────
static float* d_split_out = nullptr;
static float* d_split_m = nullptr;
static float* d_split_d = nullptr;

void execute_cuda_attention(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    const struct ggml_tensor * slot_indices,
    void * userdata,
    float * lse_out,
    const float * dense_k,
    const float * dense_v,
    const int32_t * dense_pos,
    int T_dense
) {
    CustomAttnUserData* data = static_cast<CustomAttnUserData*>(userdata);

    int n_q_heads = data->n_q_heads;
    int n_kv_heads = data->n_kv_heads;
    int D = data->D;
    int rank = data->rank;
    int S_max = data->S_max;
    int K = (slot_indices != nullptr) ? (int)slot_indices->ne[0] : 0;
    float scale = data->scale;
    int has_rope = data->has_rope ? 1 : 0;
    float rope_freq_base = data->rope_freq_base;

    // Allocate persistent scratch memory on CUDA device if needed
    if (d_split_out == nullptr) {
        cudaMalloc(&d_split_out, 64 * 8 * 128 * sizeof(float));
        cudaMalloc(&d_split_m, 64 * 8 * sizeof(float));
        cudaMalloc(&d_split_d, 64 * 8 * sizeof(float));
    }

    int S_split = (K >= 64) ? 4 : 1;

    dim3 block(64, 1, 1);
    dim3 grid(n_q_heads, S_split, 1);
    size_t shared_size = (D + 64 + D) * sizeof(float);

    // Retrieve GPU pointer mappings from GGML structures
    const float* d_Q = (const float*)Q->data;
    float* d_out = (float*)dst->data;

    // Launch main attention kernel
    decode_attention_cuda_kernel<<<grid, block, shared_size>>>(
        d_Q,
        (const int8_t*)data->kv_engine->get_U_pool()->data,
        (const float*)data->kv_engine->get_U_scale_pool()->data,
        (const half*)data->kv_engine->get_VK()->data,
        (const half*)data->kv_engine->get_VV()->data,
        (const half*)data->kv_engine->get_anchors_K()->data,
        (const half*)data->kv_engine->get_anchors_V()->data,
        (const int32_t*)data->kv_engine->get_seq_lens()->data,
        (const int32_t*)slot_indices->data,
        d_out,
        nullptr, // LSE is filled in merge pass
        n_q_heads,
        n_kv_heads,
        rank,
        S_max,
        K,
        D,
        scale,
        (const half*)data->kv_engine->get_scales()->data,
        has_rope,
        rope_freq_base,
        (const int32_t*)data->kv_engine->get_anchor_positions()->data,
        dense_k,
        dense_v,
        dense_pos,
        T_dense,
        data->approximate_attn ? 1 : 0,
        d_split_out,
        d_split_m,
        d_split_d,
        S_split
    );

    if (S_split > 1) {
        // Launch merge pass
        dim3 grid_merge(n_q_heads, 1, 1);
        dim3 block_merge(D, 1, 1);
        merge_split_k_cuda_kernel<<<grid_merge, block_merge>>>(
            d_split_out,
            d_split_m,
            d_split_d,
            d_out,
            nullptr, // LSE (CPU side copy handles it)
            S_split,
            D
        );
    }

    // Sync stream or device to ensure output is ready for host
    cudaDeviceSynchronize();
}

#endif

} // namespace diffkv
