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

// ── F7: checked CUDA calls — surface alloc/launch failures loudly instead of a
// confusing async error much later. ────────────────────────────────────────────
#define DIFFKV_CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
        std::cerr << "[DiffKV CUDA] " << #call << " failed: "                     \
                  << cudaGetErrorString(_e) << " at " << __FILE__ << ":"          \
                  << __LINE__ << std::endl;                                       \
    }                                                                             \
} while (0)

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

// ── FULL DiffKV CUDA Attention Kernel (F10) ─────────────────────────────────
// Mirrors execute_cpu_attention() (runtime/diffkv_attention.cpp), approximate_attn
// path, for the native default POOL_ROT_ABS scheme (pool pre-rotated → has_rope=0,
// no in-kernel RoPE). Per block: anchor + low-rank delta tokens (project-then-attend)
// + exact residual corrections (K score & V output), online-softmax merged across
// blocks and the dense window, split-K over blockIdx.y.
//
// Layouts (from the CPU indexing, verified against native_block_pool):
//   Q            [n_q_heads, D]
//   U_pool       int8  [n_slots, S_max, rank]              U[slot*S_max*rank + t*rank + r]
//   U_row_scale  f16   [n_slots, S_max]                    per-token dequant  [slot*S_max + t]
//   scales       f16   [n_slots]                           per-block scale blk_sc
//   VK/VV_pool   f16   [n_slots, rank, n_kv_heads, D]      [slot*rank*Hkv*D + r*Hkv*D + kv*D + d]
//   anchors_K/V  f16   [n_slots, n_kv_heads, D]            [slot*Hkv*D + kv*D + d]
//   res_*_pos    i32   [n_slots, MR]                       [slot*MR + ri]  (−1 = empty)
//   res_*_val    f16   [n_slots, MR, n_kv_heads, D]        (exact − recon) difference
// One thread per output dim d = threadIdx.x (block = D threads). Fixes the old
// block(64)-vs-D=128 bug (F11).
__global__ void diffkv_full_decode_kernel(
    const float* Q, const int8_t* U_pool, const half* VK_pool, const half* VV_pool,
    const half* anchors_K, const half* anchors_V, const int32_t* seq_lens,
    const int32_t* slot_indices, float* out_buf, float* lse_buf,
    int32_t n_q_heads, int32_t n_kv_heads, int32_t rank, int32_t S_max, int32_t K, int32_t D,
    float scale, const half* scales, const half* U_row_scale,
    const int32_t* res_K_pos, const half* res_K_val,
    const int32_t* res_V_pos, const half* res_V_val, int32_t MR,
    const float* dense_K, const float* dense_V, const int32_t* dense_positions, int32_t T_dense,
    float* split_out, float* split_m, float* split_d, int32_t S_split)
{
    const int h       = blockIdx.x;          // query head
    const int split   = blockIdx.y;          // split-K index
    const int d       = threadIdx.x;         // this thread owns output dim d
    if (h >= n_q_heads) return;
    const int g       = n_q_heads / n_kv_heads;
    const int kv_head = h / g;

    extern __shared__ float sh[];
    float* q_sh    = sh;                 // [D]
    float* qproj   = q_sh + D;           // [rank]
    float* tscore  = qproj + rank;       // [S_max]  token scores, then reused for weights
    float* wproj   = tscore + S_max;     // [rank]
    float* red     = wproj + rank;       // [<=32] reduction scratch
    __shared__ float sc_anc, sc_m, sc_l, sc_wtot;

    q_sh[d] = Q[h * D + d];
    __syncthreads();

    // running online-softmax state + unnormalised output accumulator (this dim)
    float m_i = -1e30f, l_i = 0.0f, O_d = 0.0f;

    const int blocks_per_split = (K + S_split - 1) / S_split;
    const int k0 = split * blocks_per_split;
    const int k1 = min(k0 + blocks_per_split, K);

    for (int k = k0; k < k1; ++k) {
        const int slot = slot_indices[k];
        const int slen = seq_lens[slot];
        const float blk_sc = __half2float(scales[slot]);
        const size_t vk_base = (size_t)slot * rank * n_kv_heads * D + (size_t)kv_head * D;
        const size_t ak_base = (size_t)slot * n_kv_heads * D + (size_t)kv_head * D;

        // 1) anchor score = q · anchor_K   (pool pre-rotated; no RoPE)
        float pa = q_sh[d] * __half2float(anchors_K[ak_base + d]);
        pa = blockReduceSum(pa, red);
        if (d == 0) sc_anc = pa * scale;
        __syncthreads();
        const float score_anc = sc_anc;

        // 2) q_proj[r] = q · VK[r]
        for (int r = 0; r < rank; ++r) {
            float p = q_sh[d] * __half2float(VK_pool[vk_base + (size_t)r * n_kv_heads * D + d]);
            p = blockReduceSum(p, red);
            if (d == 0) qproj[r] = p;
            __syncthreads();
        }

        // 3) residual-K score corrections: add q·resK to the token at res_K_pos[ri]
        //    (resK = exact−recon; correct-in-place, matches CPU + the ACTIVE Triton fix)
        //    Store into tscore-shift buffer keyed by token; compute per residual then scatter.
        //    We fold this into the per-token loop below via a small shared lookup.
        // 4) token scores: delta = Σ_r U[t,r]·qproj[r]; score = (delta·rowscale·blk_sc + resK + anc)·scale
        for (int t = d; t < slen; t += D) {
            const int8_t* u_row = U_pool + ((size_t)slot * S_max + t) * rank;
            float delta = 0.0f;
            for (int r = 0; r < rank; ++r) delta += (float)u_row[r] * qproj[r];
            float ku = __half2float(U_row_scale[(size_t)slot * S_max + t]);
            // residual-K: does token t carry an exact K correction?
            float res = 0.0f;
            for (int ri = 0; ri < MR; ++ri) {
                if (res_K_pos[(size_t)slot * MR + ri] == t) {
                    const half* rk = res_K_val + (((size_t)slot * MR + ri) * n_kv_heads + kv_head) * D;
                    for (int dd = 0; dd < D; ++dd) res += q_sh[dd] * __half2float(rk[dd]);
                }
            }
            tscore[t] = (delta * ku * blk_sc + res) * scale + score_anc; // == (delta*ku*blk_sc + res + anc_raw)*scale
        }
        __syncthreads();

        // 5) block max over anchor + tokens
        float local = score_anc;
        for (int t = d; t < slen; t += D) local = fmaxf(local, tscore[t]);
        local = blockReduceMax(local, red);
        if (d == 0) sc_m = local;
        __syncthreads();
        const float m_block = sc_m;

        // 6) online-softmax merge of this block into (m_i, l_i, O_d)
        const float m_new = fmaxf(m_i, m_block);
        const float resc  = expf(m_i - m_new);
        O_d *= resc;
        l_i *= resc;

        // weights (unnormalised) → reuse tscore as weights; accumulate l and w_proj
        float w_anc = expf(score_anc - m_new);
        // sum token weights + w_proj[r] = Σ_t w_t·U[t,r]·rowscale
        for (int r = d; r < rank; r += D) wproj[r] = 0.0f;
        if (d == 0) sc_l = 0.0f;
        __syncthreads();
        float l_part = 0.0f;
        for (int t = d; t < slen; t += D) {
            float w = expf(tscore[t] - m_new);
            tscore[t] = w;                 // store weight in place
            l_part += w;
        }
        l_part = blockReduceSum(l_part, red);
        if (d == 0) sc_l = l_part;
        __syncthreads();
        // w_proj[r] computed by threads r<rank over all tokens (serial in t, small)
        if (d < rank) {
            float acc = 0.0f;
            for (int t = 0; t < slen; ++t) {
                const int8_t* u_row = U_pool + ((size_t)slot * S_max + t) * rank;
                float ku = __half2float(U_row_scale[(size_t)slot * S_max + t]);
                acc += tscore[t] * (float)u_row[d] * ku;
            }
            wproj[d] = acc;
        }
        if (d == 0) sc_wtot = w_anc + sc_l;
        __syncthreads();

        // 7) accumulate this block's V into O_d (dim d):
        //    O_d += w_total·anchorV + Σ_r (wproj[r]·blk_sc)·VV[r][d] + Σ residualV
        float o = sc_wtot * __half2float(anchors_V[ak_base + d]);
        for (int r = 0; r < rank; ++r)
            o += wproj[r] * blk_sc * __half2float(VV_pool[vk_base + (size_t)r * n_kv_heads * D + d]);
        for (int ri = 0; ri < MR; ++ri) {
            int p = res_V_pos[(size_t)slot * MR + ri];
            if (p >= 0 && p < slen) {
                const half* rv = res_V_val + (((size_t)slot * MR + ri) * n_kv_heads + kv_head) * D;
                o += tscore[p] * __half2float(rv[d]);   // tscore[p] now holds w_p
            }
        }
        O_d += o;
        l_i  = l_i + sc_wtot;
        m_i  = m_new;
        __syncthreads();
    }

    // dense window (split 0 only) — exact tokens, online-merged
    if (T_dense > 0 && split == 0) {
        for (int t = 0; t < T_dense; ++t) {
            int base_k = t * n_kv_heads * D + kv_head * D;
            float s = q_sh[d] * dense_K[base_k + d];
            s = blockReduceSum(s, red);
            if (d == 0) sc_m = s * scale;
            __syncthreads();
            float sc = sc_m;
            float m_new = fmaxf(m_i, sc);
            float resc = expf(m_i - m_new);
            float w = expf(sc - m_new);
            O_d = O_d * resc + w * dense_V[t * n_kv_heads * D + kv_head * D + d];
            l_i = l_i * resc + w;
            m_i = m_new;
            __syncthreads();
        }
    }

    if (S_split == 1) {
        out_buf[h * D + d] = O_d / fmaxf(l_i, 1e-9f);
        if (d == 0 && lse_buf) lse_buf[h] = m_i + logf(fmaxf(l_i, 1e-9f));
    } else {
        split_out[(h * S_split + split) * D + d] = O_d;   // unnormalised; merge kernel divides
        if (d == 0) { split_m[h * S_split + split] = m_i; split_d[h * S_split + split] = l_i; }
    }
}

// ── CUDA Attention Kernel (legacy anchor+dense stub — kept for A/B) ──────────
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
static int    d_split_units = 0;   // capacity in (n_q_heads * S_split) units
static int    d_split_D     = 0;   // head_dim the split_out buffer was sized for

void execute_cuda_attention(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    const int32_t * slot_indices,
    int            actual_K,
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
    int K = actual_K;
    float scale = data->scale;
    int has_rope = data->has_rope ? 1 : 0;
    float rope_freq_base = data->rope_freq_base;

    // F6/F7/F9: size split-K scratch from the ACTUAL model dims. Was hardcoded
    // 64*8*128 → out-of-bounds writes for models with >64 query heads or head_dim
    // >128. S_split is at most 4 (see below); size for that max, and reallocate
    // (freeing the old buffers) whenever the model dims grow.
    const int S_SPLIT_MAX = 4;
    int needed_units = n_q_heads * S_SPLIT_MAX;
    if (d_split_out == nullptr || d_split_units < needed_units || d_split_D != D) {
        if (d_split_out) { cudaFree(d_split_out); cudaFree(d_split_m); cudaFree(d_split_d); }
        DIFFKV_CUDA_CHECK(cudaMalloc(&d_split_out, (size_t)needed_units * D * sizeof(float)));
        DIFFKV_CUDA_CHECK(cudaMalloc(&d_split_m,   (size_t)needed_units * sizeof(float)));
        DIFFKV_CUDA_CHECK(cudaMalloc(&d_split_d,   (size_t)needed_units * sizeof(float)));
        d_split_units = needed_units;
        d_split_D = D;
    }

    // Allocate and copy slot indices to CUDA scratch memory
    static int32_t* d_slot_indices_scratch = nullptr;
    static int scratch_capacity = 0;
    if (K > 0 && slot_indices != nullptr) {
        if (d_slot_indices_scratch == nullptr || scratch_capacity < K) {
            if (d_slot_indices_scratch) cudaFree(d_slot_indices_scratch);
            DIFFKV_CUDA_CHECK(cudaMalloc(&d_slot_indices_scratch, std::max(256, K) * sizeof(int32_t)));
            scratch_capacity = std::max(256, K);
        }
        DIFFKV_CUDA_CHECK(cudaMemcpy(d_slot_indices_scratch, slot_indices, K * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    int S_split = (K >= 64) ? 4 : 1;

    dim3 grid(n_q_heads, S_split, 1);
    const float* d_Q = (const float*)Q->data;
    float* d_out = (float*)dst->data;

    // F10: full DiffKV kernel (anchor + low-rank deltas + exact residuals + dense),
    // one thread per output dim. Legacy anchor+dense stub via DIFFKV_CUDA_ANCHOR_ONLY=1
    // for A/B. Correctness fallback is the CPU path (DIFFKV_FORCE_CPU_ATTN=1).
    const char* anchor_only_env = std::getenv("DIFFKV_CUDA_ANCHOR_ONLY");
    if (!(anchor_only_env && std::string(anchor_only_env) == "1")) {
        dim3 block(D, 1, 1);
        size_t shared_size = ((size_t)D + rank + S_max + rank + 32) * sizeof(float);
        diffkv_full_decode_kernel<<<grid, block, shared_size>>>(
            d_Q,
            (const int8_t*)data->kv_engine->get_U_pool()->data,
            (const half*)data->kv_engine->get_VK()->data,
            (const half*)data->kv_engine->get_VV()->data,
            (const half*)data->kv_engine->get_anchors_K()->data,
            (const half*)data->kv_engine->get_anchors_V()->data,
            (const int32_t*)data->kv_engine->get_seq_lens()->data,
            d_slot_indices_scratch, d_out, nullptr,
            n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
            (const half*)data->kv_engine->get_scales()->data,
            (const half*)data->kv_engine->get_U_row_scale()->data,
            (const int32_t*)data->kv_engine->get_res_K_pos()->data,
            (const half*)data->kv_engine->get_res_K_val()->data,
            (const int32_t*)data->kv_engine->get_res_V_pos()->data,
            (const half*)data->kv_engine->get_res_V_val()->data,
            NativeBlockPool::MAX_RESIDUAL,
            dense_k, dense_v, dense_pos, T_dense,
            d_split_out, d_split_m, d_split_d, S_split);
        DIFFKV_CUDA_CHECK(cudaGetLastError());
    } else {
        dim3 block(64, 1, 1);
        size_t shared_size = (D + 64 + D) * sizeof(float);
        decode_attention_cuda_kernel<<<grid, block, shared_size>>>(
            d_Q,
            (const int8_t*)data->kv_engine->get_U_pool()->data,
            (const float*)data->kv_engine->get_U_scale_pool()->data,
            (const half*)data->kv_engine->get_VK()->data,
            (const half*)data->kv_engine->get_VV()->data,
            (const half*)data->kv_engine->get_anchors_K()->data,
            (const half*)data->kv_engine->get_anchors_V()->data,
            (const int32_t*)data->kv_engine->get_seq_lens()->data,
            d_slot_indices_scratch, d_out, nullptr,
            n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
            (const half*)data->kv_engine->get_scales()->data,
            has_rope, rope_freq_base,
            (const int32_t*)data->kv_engine->get_anchor_positions()->data,
            dense_k, dense_v, dense_pos, T_dense,
            data->approximate_attn ? 1 : 0,
            d_split_out, d_split_m, d_split_d, S_split);
        DIFFKV_CUDA_CHECK(cudaGetLastError());
    }

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
        DIFFKV_CUDA_CHECK(cudaGetLastError());
    }

    // F8: the host reads d_out immediately after this call, so a sync is required —
    // but only for OUR work. All launches above go to the legacy default stream, so
    // cudaStreamSynchronize(0) is the minimal correct barrier; the previous
    // cudaDeviceSynchronize() stalled EVERY stream on the device once per decode
    // token (a full device barrier per token — audit finding F8). Going fully async
    // (sync only when the caller consumes dst) needs the ggml-stream integration and
    // is out of scope for a blind fix; see CUDA_TRITON_AUDIT.md checklist C5.
    DIFFKV_CUDA_CHECK(cudaStreamSynchronize(0));
}

#endif

} // namespace diffkv
