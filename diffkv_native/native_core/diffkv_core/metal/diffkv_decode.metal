#include <metal_stdlib>
using namespace metal;

// Helper struct to merge two online softmax states
struct SoftmaxState {
    float m; // max score
    float d; // sum of exp(score - max)
};

inline SoftmaxState merge_softmax_states(SoftmaxState a, SoftmaxState b) {
    if (a.m > b.m) {
        return { a.m, a.d + b.d * exp(b.m - a.m) };
    } else {
        return { b.m, b.d + a.d * exp(a.m - b.m) };
    }
}

// Fused Project-Then-Attend Metal decode attention kernel.
// Ported 1:1 from ACTIVE_RUNTIME/native_core/diffkv_core/metal/diffkv_decode.metal
//
// Algorithm (per query head threadgroup):
//   PASS 1:
//     For each block k:
//       1. Compute anchor dot product s_anc = q · rotated_anchor_K * scale
//       2. Cache Q projection q_proj[r] = q · rotated_VK[r] for r in rank
//       3. Delta scores: s_t = (q_proj · U_k[t]) * scale_u * block_scale + s_anc
//       4. Online softmax over (s_anc, s_0, s_1, ..., s_{slen-1})
//   PASS 2:
//     For each block k:
//       1. Load cached anchor score and q_proj
//       2. w_anc = exp(s_anc - global_m) / global_d
//       3. For each delta t: w_t = exp(s_t - global_m) / global_d
//          Accumulate: sum_w += w_t, w_proj[r] += w_t * U_k[t,r] * scale_u
//       4. thread_val[d] += (w_anc + sum_w) * anchor_V[d]
//                        + w_proj @ VV * block_scale
//
kernel void decode_attention_metal_kernel(
    device const float* Q [[buffer(0)]],                   // [H_q, D] — F32 from ggml CPU backend
    device const int8_t* U_pool [[buffer(1)]],            // [N_pool, S_max, R] int8
    device const half* U_scale_pool [[buffer(2)]],        // [N_pool] float16
    device const half* VK_pool [[buffer(3)]],             // [N_pool, R, n_kv, D] float16
    device const half* VV_pool [[buffer(4)]],             // [N_pool, R, n_kv, D] float16
    device const half* anchors_K [[buffer(5)]],           // [N_pool, n_kv, D] float16
    device const half* anchors_V [[buffer(6)]],           // [N_pool, n_kv, D] float16
    device const int32_t* seq_lens [[buffer(7)]],         // [N_pool] int32
    device const int32_t* slot_indices [[buffer(8)]],     // [K] int32
    device float* out_buf [[buffer(9)]],                  // [H_q, D] float32 output
    device float* lse_buf [[buffer(10)]],                 // [H_q] float32 log-sum-exp

    // Uniform scalars
    device const int32_t& n_q_heads [[buffer(11)]],
    device const int32_t& n_kv_heads [[buffer(12)]],
    device const int32_t& rank [[buffer(13)]],
    device const int32_t& S_max [[buffer(14)]],
    device const int32_t& K [[buffer(15)]],
    device const int32_t& D [[buffer(16)]],
    device const float& scale [[buffer(17)]],
    device const half* scales [[buffer(18)]],             // [N_pool] float16 block scales
    device const int32_t& has_rope [[buffer(19)]],
    device const float& rope_freq_base [[buffer(20)]],    
    device const int32_t* anchor_positions [[buffer(21)]], // [N_pool] actual sequence positions

    uint tg_idx [[threadgroup_position_in_grid]],  // query head index 0..H_q-1
    uint tid [[thread_position_in_threadgroup]],
    uint t_per_tg [[threads_per_threadgroup]]
) {
    if (tg_idx >= (uint)n_q_heads) return;

    // GQA: which kv_head this query head reads from
    const int g = n_q_heads / n_kv_heads;
    const int kv_head = (int)tg_idx / g;
    const int half_d = D / 2;

    // ── Shared memory ─────────────────────────────────────────────────────────
    threadgroup float q_shared[128];       // cached query [D], D <= 128

    // Reduction buffers (threadgroup size = 64)
    threadgroup float red_m[64];
    threadgroup float red_d[64];
    threadgroup float red_w_proj[32];      // reduced w_proj per rank
    threadgroup float red_sum_w;
    // Reduction scratch: [64, 32] = 8192 bytes
    threadgroup float red_proj_temp[64 * 32];

    // Cache for anchor scores across K blocks — cap at 128
    threadgroup float scores_anc_cached[128];

    // Shared buffer for rotated anchor key [D]
    threadgroup float ak_rot_shared[128];

    // Value accumulator (thread-local, max D = 128)
    thread float thread_val[128] = { 0.0f };

    // 1. Cache the query into shared memory
    for (int d = (int)tid; d < D; d += (int)t_per_tg) {
        q_shared[d] = (float)Q[tg_idx * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── PASS 1: Online softmax (max + denominator) ────────────────────────────
    SoftmaxState sm_state = { -1e30f, 0.0f };

    for (int k = 0; k < K; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];
        int anchor_pos = anchor_positions[slot_id];

        // ── Step 1a: Compute rotated anchor key and store in shared mem ───────
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            float raw_ak = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d];
            if (has_rope) {
                int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                int idx = (d < half_d) ? d : (d - half_d);
                float theta = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                float angle = anchor_pos * theta;
                float c = cos(angle);
                float s = sin(angle);
                float raw_partner = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner];
                float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                ak_rot_shared[d] = raw_ak * c + rot_contrib * s;
            } else {
                ak_rot_shared[d] = raw_ak;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Step 1b: Anchor dot product (parallel reduction over D) ───────────
        float thread_anc_sum = 0.0f;
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            thread_anc_sum += q_shared[d] * ak_rot_shared[d];
        }
        red_m[tid] = thread_anc_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) red_m[tid] += red_m[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score_anc = red_m[0];

        // Cache anchor score
        if (tid == 0 && k < 128) scores_anc_cached[k] = score_anc;

        // ── Step 1c: Anchor score → online softmax ────────────────────────────
        if (tid == 0) {
            float s_anc_scaled = score_anc * scale;
            sm_state = merge_softmax_states(sm_state, { s_anc_scaled, 1.0f });
        }

        // ── Step 1d: Delta scores via exact key reconstruction + per-token RoPE ──
        float scale_u = (float)U_scale_pool[slot_id];
        float block_scale = (float)scales[slot_id];
        for (int t = (int)tid; t < slen; t += (int)t_per_tg) {
            float t_score_sum = 0.0f;
            int u_offset = slot_id * S_max * rank + t * rank;
            for (int d = 0; d < half_d; ++d) {
                // Partner 1
                float raw_ak_1 = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d];
                float sum_r_vk_1 = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    sum_r_vk_1 += (float)U_pool[u_offset + r] * (float)VK_pool[base_vk_offset + d];
                }
                float k_unrot_1 = raw_ak_1 + block_scale * scale_u * sum_r_vk_1;

                // Partner 2
                int d2 = d + half_d;
                float raw_ak_2 = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d2];
                float sum_r_vk_2 = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    sum_r_vk_2 += (float)U_pool[u_offset + r] * (float)VK_pool[base_vk_offset + d2];
                }
                float k_unrot_2 = raw_ak_2 + block_scale * scale_u * sum_r_vk_2;

                float k_rot_1 = k_unrot_1;
                float k_rot_2 = k_unrot_2;
                if (has_rope) {
                    float theta = 1.0f / pow(rope_freq_base, (2.0f * d) / D);
                    int pos = anchor_pos + 1 + t;
                    float angle = pos * theta;
                    float c = cos(angle);
                    float s = sin(angle);
                    k_rot_1 = k_unrot_1 * c - k_unrot_2 * s;
                    k_rot_2 = k_unrot_2 * c + k_unrot_1 * s;
                }
                t_score_sum += q_shared[d] * k_rot_1 + q_shared[d2] * k_rot_2;
            }
            float t_score = t_score_sum * scale;
            sm_state = merge_softmax_states(sm_state, { t_score, 1.0f });
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce softmax states across all threads in threadgroup
    red_m[tid] = sm_state.m;
    red_d[tid] = sm_state.d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            SoftmaxState s1 = { red_m[tid], red_d[tid] };
            SoftmaxState s2 = { red_m[tid + stride], red_d[tid + stride] };
            SoftmaxState merged = merge_softmax_states(s1, s2);
            red_m[tid] = merged.m;
            red_d[tid] = merged.d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_m = red_m[0];
    float global_d = red_d[0];

    if (tid == 0) {
        lse_buf[tg_idx] = global_m + log(max(global_d, 1e-9f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── PASS 2: Accumulate values using Project-Then-Attend ──────────────────
    for (int k = 0; k < K; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];
        float scale_u = (float)U_scale_pool[slot_id];
        float block_scale = (float)scales[slot_id];

        // Reload anchor score from cache or recompute
        float score_anc;
        if (k < 128) {
            score_anc = scores_anc_cached[k];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Recompute: rotate anchor key (k >= 128, rare)
            for (int d = (int)tid; d < D; d += (int)t_per_tg) {
                float raw_ak = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d];
                if (has_rope) {
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    int idx = (d < half_d) ? d : (d - half_d);
                    float theta = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                    int anchor_pos = anchor_positions[slot_id];
                    float angle = anchor_pos * theta;
                    float c = cos(angle);
                    float s = sin(angle);
                    float raw_partner = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner];
                    float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                    ak_rot_shared[d] = raw_ak * c + rot_contrib * s;
                } else {
                    ak_rot_shared[d] = raw_ak;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float thread_anc = 0.0f;
            for (int d = (int)tid; d < D; d += (int)t_per_tg) thread_anc += q_shared[d] * ak_rot_shared[d];
            red_m[tid] = thread_anc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
                if (tid < stride) red_m[tid] += red_m[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            score_anc = red_m[0];
        }

        // Anchor attention weight
        float w_anc = exp(score_anc * scale - global_m) / max(global_d, 1e-9f);

        // Delta weights: per-thread partial sums of w_t and w_t * U[t,r] * scale_u
        float local_sum_w = 0.0f;
        float local_w_proj[32] = { 0.0f };  // per-rank weighted U accumulator

        int anchor_pos = anchor_positions[slot_id];
        for (int t = (int)tid; t < slen; t += (int)t_per_tg) {
            float t_score_sum = 0.0f;
            int u_offset = slot_id * S_max * rank + t * rank;
            for (int d = 0; d < half_d; ++d) {
                // Partner 1
                float raw_ak_1 = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d];
                float sum_r_vk_1 = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    sum_r_vk_1 += (float)U_pool[u_offset + r] * (float)VK_pool[base_vk_offset + d];
                }
                float k_unrot_1 = raw_ak_1 + block_scale * scale_u * sum_r_vk_1;

                // Partner 2
                int d2 = d + half_d;
                float raw_ak_2 = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d2];
                float sum_r_vk_2 = 0.0f;
                for (int r = 0; r < rank; ++r) {
                    int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    sum_r_vk_2 += (float)U_pool[u_offset + r] * (float)VK_pool[base_vk_offset + d2];
                }
                float k_unrot_2 = raw_ak_2 + block_scale * scale_u * sum_r_vk_2;

                float k_rot_1 = k_unrot_1;
                float k_rot_2 = k_unrot_2;
                if (has_rope) {
                    float theta = 1.0f / pow(rope_freq_base, (2.0f * d) / D);
                    int pos = anchor_pos + 1 + t;
                    float angle = pos * theta;
                    float c = cos(angle);
                    float s = sin(angle);
                    k_rot_1 = k_unrot_1 * c - k_unrot_2 * s;
                    k_rot_2 = k_unrot_2 * c + k_unrot_1 * s;
                }
                t_score_sum += q_shared[d] * k_rot_1 + q_shared[d2] * k_rot_2;
            }
            float t_score = t_score_sum * scale;
            float w_t = exp(t_score - global_m) / max(global_d, 1e-9f);
            local_sum_w += w_t;
            for (int r = 0; r < rank; ++r) {
                local_w_proj[r] += w_t * (float)U_pool[u_offset + r] * scale_u;
            }
        }

        // Reduce sum_w over threadgroup
        red_m[tid] = local_sum_w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) red_m[tid] += red_m[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) red_sum_w = red_m[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce w_proj[r] over all 64 threads in threadgroup
        for (int r = 0; r < rank; ++r) {
            red_proj_temp[(int)tid * rank + r] = local_w_proj[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < (uint)rank) {
            float sum_proj = 0.0f;
            for (int t = 0; t < (int)t_per_tg; ++t) sum_proj += red_proj_temp[t * rank + (int)tid];
            red_w_proj[tid] = sum_proj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate block value contribution to thread_val
        // thread_val[d] += (w_anc + sum_w_delta) * anchor_V[d]
        //               + (w_proj @ VV) * block_scale
        float w_total_anc = w_anc + red_sum_w;
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            thread_val[d] += w_total_anc * (float)anchors_V[slot_id * n_kv_heads * D + kv_head * D + d];

            float svd_v = 0.0f;
            int base_vv_offset = slot_id * rank * n_kv_heads * D + kv_head * D + d;
            for (int r = 0; r < rank; ++r) {
                svd_v += red_w_proj[r] * (float)VV_pool[base_vv_offset + r * n_kv_heads * D];
            }
            thread_val[d] += svd_v * block_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (int d = (int)tid; d < D; d += (int)t_per_tg) {
        out_buf[tg_idx * D + d] = thread_val[d];
    }
}
