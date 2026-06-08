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
// Parallelized as 1 Threadgroup per Query Head.
kernel void decode_attention_metal_kernel(
    device const half* Q [[buffer(0)]],                  // [H_q, D]
    device const int8_t* U_pool [[buffer(1)]],            // [N_pool, S_max, R]
    device const half* U_scale_pool [[buffer(2)]],       // [N_pool]
    device const half* VK_pool [[buffer(3)]],            // [N_pool, R, n_kv, D]
    device const half* VV_pool [[buffer(4)]],            // [N_pool, R, n_kv, D]
    device const half* anchors_K [[buffer(5)]],          // [N_pool, n_kv, D]
    device const half* anchors_V [[buffer(6)]],          // [N_pool, n_kv, D]
    device const int32_t* seq_lens [[buffer(7)]],        // [N_pool]
    device const int32_t* slot_indices [[buffer(8)]],    // [K]
    device half* out_buf [[buffer(9)]],                  // [H_q, D]
    device float* lse_buf [[buffer(10)]],                // [H_q]
    
    // Uniform configurations
    device const int32_t& n_q_heads [[buffer(11)]],
    device const int32_t& n_kv_heads [[buffer(12)]],
    device const int32_t& rank [[buffer(13)]],
    device const int32_t& S_max [[buffer(14)]],
    device const int32_t& K [[buffer(15)]],
    device const int32_t& D [[buffer(16)]],
    device const float& scale [[buffer(17)]],
    device const half* scales [[buffer(18)]],            // [N_pool]
    // RoPE buffers: per-slot anchor cosine and sine [K, D] float32
    device const float* cos_anc [[buffer(19)]],          // [K, D] float32
    device const float* sin_anc [[buffer(20)]],          // [K, D] float32
    device const int32_t& has_rope [[buffer(21)]],       // 1 if RoPE should be applied

    uint tg_idx [[threadgroup_position_in_grid]],       // Query head index (0..H_q-1)
    uint tid [[thread_position_in_threadgroup]],        // Thread index within threadgroup
    uint t_per_tg [[threads_per_threadgroup]]           // Size of threadgroup
) {
    // Return early if threadgroup is out of bounds
    if (tg_idx >= (uint)n_q_heads) return;

    // GQA parameters
    const int g = n_q_heads / n_kv_heads;
    const int kv_head = tg_idx / g;

    // ── Shared memory allocations ─────────────────────────────────────────────
    // Support up to D = 128, Rank = 32
    threadgroup float q_shared[128];
    threadgroup float q_proj_shared[32];
    
    // Shared buffers for reductions
    threadgroup float red_m[128]; // Max threadgroup size 128
    threadgroup float red_d[128];
    threadgroup float red_w_proj[32];
    threadgroup float red_sum_w;
    threadgroup float red_proj_temp[64 * 32]; // [threads_per_tg, rank] temp buffer
    threadgroup float scores_anc_cached[128]; // cache for anchor scores (up to 128 blocks)
    threadgroup float q_proj_cached[128 * 32]; // cache for query projections (up to 128 blocks)
    // Shared buffer to hold rotated anchor key for current block [D]
    threadgroup float ak_rot_shared[128];

    // 1. Cache the query vector in shared memory
    for (int d = tid; d < D; d += t_per_tg) {
        q_shared[d] = (float)Q[tg_idx * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── PASS 1: Compute online softmax maximum and denominator ────────────────
    SoftmaxState sm_state = { -1e30f, 0.0f };
    const int half_d = D / 2;

    for (int k = 0; k < K; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];

        // 1. Compute anchor dot product score (using all threads to collaborate)
        for (int d = tid; d < D; d += t_per_tg) {
            float raw_ak = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d];
            if (has_rope) {
                float c = cos_anc[k * D + d];
                float s = sin_anc[k * D + d];
                int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                float raw_partner = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner];
                float rot_partner_contrib = (d < half_d) ? -raw_partner : raw_partner;
                ak_rot_shared[d] = raw_ak * c + rot_partner_contrib * s;
            } else {
                ak_rot_shared[d] = raw_ak;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float thread_anc_sum = 0.0f;
        for (int d = tid; d < D; d += t_per_tg) {
            thread_anc_sum += q_shared[d] * ak_rot_shared[d];
        }
        
        // Reduction over threadgroup to get final anchor dot product
        red_m[tid] = thread_anc_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                red_m[tid] += red_m[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score_anc = red_m[0];
        if (k < 128 && tid == 0) {
            scores_anc_cached[k] = score_anc;
        }
        
        // 2. Compute query projection q_proj[r] = dot(q, VK_pool[slot_id, r])
        if (tid < (uint)rank) {
            float proj_val = 0.0f;
            int base_vk_offset = slot_id * rank * n_kv_heads * D + tid * n_kv_heads * D + kv_head * D;
            for (int d = 0; d < D; ++d) {
                float raw_vk = (float)VK_pool[base_vk_offset + d];
                float vk_rot;
                if (has_rope) {
                    float c = cos_anc[k * D + d];
                    float s = sin_anc[k * D + d];
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    float raw_vk_partner = (float)VK_pool[base_vk_offset + partner];
                    float rot_partner_contrib = (d < half_d) ? -raw_vk_partner : raw_vk_partner;
                    vk_rot = raw_vk * c + rot_partner_contrib * s;
                } else {
                    vk_rot = raw_vk;
                }
                proj_val += q_shared[d] * vk_rot;
            }
            q_proj_shared[tid] = proj_val;
            
            if (k < 128) {
                q_proj_cached[k * rank + tid] = proj_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Process anchor token
        // Let thread 0 handle the anchor score update
        if (tid == 0) {
            float s_anc_scaled = score_anc * scale;
            sm_state = merge_softmax_states(sm_state, { s_anc_scaled, 1.0f });
        }

        // 4. Process reconstructed tokens
        float scale_u = (float)U_scale_pool[slot_id];
        float block_scale = (float)scales[slot_id];
        for (int t = tid; t < slen; t += t_per_tg) {
            float delta_sum = 0.0f;
            int u_offset = slot_id * S_max * rank + t * rank;
            for (int r = 0; r < rank; ++r) {
                delta_sum += q_proj_shared[r] * (float)U_pool[u_offset + r];
            }
            float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
            sm_state = merge_softmax_states(sm_state, { t_score, 1.0f });
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce local softmax states of all threads in the threadgroup
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

    // Write Log-Sum-Exp to global output buffer
    if (tid == 0) {
        lse_buf[tg_idx] = global_m + log(max(global_d, 1e-9f));
    }

    // ── PASS 2: Accumulate values ─────────────────────────────────────────────
    // Thread-local value accumulator
    float thread_val[128] = { 0.0f }; // Supports D up to 128

    for (int k = 0; k < K; ++k) {
        int slot_id = slot_indices[k];
        int slen = seq_lens[slot_id];

        // 1. Recompute anchor score & query projection (or load from cache)
        float score_anc;
        if (k < 128) {
            score_anc = scores_anc_cached[k];
            if (tid < (uint)rank) {
                q_proj_shared[tid] = q_proj_cached[k * rank + tid];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Recompute using RoPE-rotated anchor keys
            for (int d = tid; d < D; d += t_per_tg) {
                float raw_ak = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + d];
                if (has_rope) {
                    float c = cos_anc[k * D + d];
                    float s = sin_anc[k * D + d];
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    float raw_partner = (float)anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner];
                    float rot_partner_contrib = (d < half_d) ? -raw_partner : raw_partner;
                    ak_rot_shared[d] = raw_ak * c + rot_partner_contrib * s;
                } else {
                    ak_rot_shared[d] = raw_ak;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float thread_anc_sum = 0.0f;
            for (int d = tid; d < D; d += t_per_tg) {
                thread_anc_sum += q_shared[d] * ak_rot_shared[d];
            }
            red_m[tid] = thread_anc_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    red_m[tid] += red_m[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            score_anc = red_m[0];

            if (tid < (uint)rank) {
                float proj_val = 0.0f;
                int base_vk_offset = slot_id * rank * n_kv_heads * D + tid * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float raw_vk = (float)VK_pool[base_vk_offset + d];
                    float vk_rot;
                    if (has_rope) {
                        float c = cos_anc[k * D + d];
                        float s = sin_anc[k * D + d];
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_vk_partner = (float)VK_pool[base_vk_offset + partner];
                        float rot_partner_contrib = (d < half_d) ? -raw_vk_partner : raw_vk_partner;
                        vk_rot = raw_vk * c + rot_partner_contrib * s;
                    } else {
                        vk_rot = raw_vk;
                    }
                    proj_val += q_shared[d] * vk_rot;
                }
                q_proj_shared[tid] = proj_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 2. Compute local sum of weights and projected weights for deltas
        float w_anc = exp(score_anc * scale - global_m) / max(global_d, 1e-9f);
        float local_sum_w = 0.0f;
        float local_w_proj[32] = { 0.0f }; // Support up to Rank = 32
        
        float scale_u = (float)U_scale_pool[slot_id];
        float block_scale = (float)scales[slot_id];
        for (int t = tid; t < slen; t += t_per_tg) {
            float delta_sum = 0.0f;
            int u_offset = slot_id * S_max * rank + t * rank;
            for (int r = 0; r < rank; ++r) {
                delta_sum += q_proj_shared[r] * (float)U_pool[u_offset + r];
            }
            float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
            float w_t = exp(t_score - global_m) / max(global_d, 1e-9f);
            
            local_sum_w += w_t;
            for (int r = 0; r < rank; ++r) {
                local_w_proj[r] += w_t * (float)U_pool[u_offset + r] * scale_u;
            }
        }

        // Reduce sum of weights and projected weights over threadgroup
        // Reduce sum_w
        red_m[tid] = local_sum_w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = t_per_tg / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                red_m[tid] += red_m[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            red_sum_w = red_m[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce w_proj[r] for all r in parallel using the shared temp buffer
        for (int r = 0; r < rank; ++r) {
            red_proj_temp[tid * rank + r] = local_w_proj[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < (uint)rank) {
            float sum_proj = 0.0f;
            for (int t = 0; t < (int)t_per_tg; ++t) {
                sum_proj += red_proj_temp[t * rank + tid];
            }
            red_w_proj[tid] = sum_proj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Add block value contributions to thread_val
        float w_total_anc = w_anc + red_sum_w;
        for (int d = tid; d < D; d += t_per_tg) {
            // Anchor component (base for all tokens in block)
            thread_val[d] += w_total_anc * (float)anchors_V[slot_id * n_kv_heads * D + kv_head * D + d];

            // SVD basis component
            float svd_v_contribution = 0.0f;
            int base_vv_offset = slot_id * rank * n_kv_heads * D + kv_head * D + d;
            for (int r = 0; r < rank; ++r) {
                svd_v_contribution += red_w_proj[r] * (float)VV_pool[base_vv_offset + r * n_kv_heads * D];
            }
            thread_val[d] += svd_v_contribution * block_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final output values to out_buf
    for (int d = tid; d < D; d += t_per_tg) {
        out_buf[tg_idx * D + d] = (half)thread_val[d];
    }
}
