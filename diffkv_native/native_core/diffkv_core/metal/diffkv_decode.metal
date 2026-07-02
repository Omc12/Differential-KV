#include <metal_stdlib>
using namespace metal;

inline float reduce_sum_tg64(float val, threadgroup float* red_shared, uint tid) {
    float simd_s = simd_sum(val);
    if ((tid & 31) == 0) {
        red_shared[tid >> 5] = simd_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red_shared[0] + red_shared[1];
}

inline float reduce_max_tg64(float val, threadgroup float* red_shared, uint tid) {
    float simd_m = simd_max(val);
    if ((tid & 31) == 0) {
        red_shared[tid >> 5] = simd_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return max(red_shared[0], red_shared[1]);
}

// Helper struct to merge two online softmax states
struct SoftmaxState {
    float m; // max score
    float d; // sum of exp(score - max)
};

inline SoftmaxState merge_softmax_states(SoftmaxState a, SoftmaxState b) {
    if (a.d == 0.0f) return b;
    if (b.d == 0.0f) return a;
    if (a.m > b.m) {
        return { a.m, a.d + b.d * exp(b.m - a.m) };
    } else {
        return { b.m, b.d + a.d * exp(a.m - b.m) };
    }
}

// Unified DiffKV decode attention kernel.
//
// Combines:
//   1. Sparse compressed blocks via Project-Then-Attend (q_proj_cached optimization
//      ported from ACTIVE_RUNTIME): avoids double key-reconstruction across PASS 1 & 2.
//      Uses anchor-position RoPE for VK (slight approximation, same as Python reference).
//   2. Dense window tokens with exact per-token RoPE (buffers 22-25).
//
// Outputs fully combined sparse + dense result directly into out_buf.
// No CPU-side attention or LSE combine required after this call.
//
// Shared memory budget (~20 KB, well within Apple Silicon 32 KB limit):
//   q_shared[128]         =  512 B
//   q_proj_shared[32]     =  128 B
//   red_m/d[64]           =  512 B
//   red_w_proj[32]        =  128 B
//   red_proj_temp[64*32]  = 8192 B
//   scores_anc_cached[64] =  256 B
//   q_proj_cached[64*32]  = 8192 B   ← NEW (Project-Then-Attend cache)
//   ak_rot_shared[128]    =  512 B
//   dense_weights[2048]    = 8192 B   dense window weights (matches native_maxd=2048)
//   Total                 ≈ 26.4 KB
struct AttentionParams {
    int32_t n_q_heads;
    int32_t n_kv_heads;
    int32_t rank;
    int32_t S_max;
    int32_t K;
    int32_t D;
    float scale;
    int32_t has_rope;
    float rope_freq_base;
    int32_t T_dense;
    int32_t approximate_attn;
    int32_t S_split;
    int32_t max_residual;
    // 1 → pool K (anchors, VK basis, residual K) is stored FULLY rotated at its
    // absolute sequence position (POOL_ROT_ABS ingest). Pool score sites must then
    // apply NO rotation; only the dense window (raw K + positions) still rotates.
    int32_t pool_prerotated;
    int32_t is_q8;
};

typedef struct {
    half d;
    int8_t qs[32];
} block_q8_0;

kernel void decode_attention_metal_kernel(
    // ── Sparse pool buffers (unchanged from original) ─────────────────────────
    device const float*   Q             [[buffer(0)]],   // [H_q, D] F32
    device const int8_t*  U_pool        [[buffer(1)]],   // [N_pool, S_max, R] int8
    device const half*    U_row_scale_pool [[buffer(2)]], // [N_pool, S_max] f16
    device const void*    VK_pool       [[buffer(3)]],   // [N_pool, R, n_kv, D] f16
    device const void*    VV_pool       [[buffer(4)]],   // [N_pool, R, n_kv, D] f16
    device const void*    anchors_K     [[buffer(5)]],   // [N_pool, n_kv, D] f16
    device const void*    anchors_V     [[buffer(6)]],   // [N_pool, n_kv, D] f16
    device const int32_t* seq_lens      [[buffer(7)]],   // [N_pool] int32
    device const int32_t* slot_indices  [[buffer(8)]],   // [K] int32
    device float*         out_buf       [[buffer(9)]],   // [H_q, D] f32 combined output
    device float*         lse_buf       [[buffer(10)]],  // [H_q] f32 log-sum-exp
    device const AttentionParams& params [[buffer(11)]],
    device const half*    scales        [[buffer(12)]],  // [N_pool] f16 block scales
    device const int32_t* anchor_positions[[buffer(13)]],// [N_pool] sequence positions
    // ── Dense window buffers (NEW) ─────────────────────────────────────────────
    device const float*   dense_K       [[buffer(14)]],  // [T_dense, n_kv, D] f32
    device const float*   dense_V       [[buffer(15)]],  // [T_dense, n_kv, D] f32
    device const int32_t* dense_positions[[buffer(16)]], // [T_dense] sequence positions
    device float*         split_out     [[buffer(17)]],
    device float*         split_m       [[buffer(18)]],
    device float*         split_d       [[buffer(19)]],
    device const int32_t* res_K_pos     [[buffer(20)]],
    device const int32_t* res_V_pos     [[buffer(21)]],
    device const void*    res_K_val     [[buffer(22)]],
    device const void*    res_V_val     [[buffer(23)]],

    uint3 tg_idx   [[threadgroup_position_in_grid]], // tg_idx.x is head, tg_idx.y is split index
    uint3 tid_vec  [[thread_position_in_threadgroup]],
    uint3 t_per_tg_vec [[threads_per_threadgroup]]
) {
#define READ_VAL(buf, idx) ((params.is_q8) ? ((float)((device const block_q8_0 *)(buf))[(idx) / 32].d * (float)((device const block_q8_0 *)(buf))[(idx) / 32].qs[(idx) % 32]) : (float)((device const half *)(buf))[(idx)])
    const uint tid = tid_vec.x;
    const uint t_per_tg = t_per_tg_vec.x;
    const int32_t n_q_heads = params.n_q_heads;
    const int32_t n_kv_heads = params.n_kv_heads;
    const int32_t rank = params.rank;
    const int32_t S_max = params.S_max;
    const int32_t K = params.K;
    const int32_t D = params.D;
    const float scale = params.scale;
    const int32_t has_rope = params.has_rope;
    const float rope_freq_base = params.rope_freq_base;
    const int32_t T_dense = params.T_dense;
    const int32_t approximate_attn = params.approximate_attn;
    const int32_t S_split = params.S_split;
    const int32_t max_residual = params.max_residual;
    // Rotation gate for POOL content only (anchors / VK / per-token recon / residual K).
    // Dense-window sites keep using has_rope: dense K is stored raw with positions.
    const int32_t rope_pool = has_rope && !params.pool_prerotated;

    if (tg_idx.x >= (uint)n_q_heads) return;

    const int g        = n_q_heads / n_kv_heads;
    const int kv_head  = (int)tg_idx.x / g;
    const int half_d   = D / 2;

    // ── Shared memory ─────────────────────────────────────────────────────────
    threadgroup float q_shared[128];          // Query cache [D ≤ 128]
    threadgroup float q_proj_shared[64];      // Q · VK_rot projection [rank ≤ 64]
    threadgroup float red_m[64];              // General-purpose reduction buffer
    threadgroup float red_d[64];              // Softmax-d reduction buffer
    threadgroup float red_w_proj[64];         // Reduced w_proj[r]
    threadgroup float red_sum_w;              // Sum of token delta weights
    threadgroup float red_proj_temp[64 * 64]; // [threads × rank] temp for w_proj reduce
    threadgroup float scores_cached[64 * 32]; // Unified token scores cache (≤ 64 blocks × max 32 tokens/block)
    threadgroup float ak_rot_shared[128];     // Rotated anchor key [D]
    threadgroup float dense_weights[1024];    // Dense window weights — chunked 1024 at a time, so 1024 suffices for any T_dense
    threadgroup float w_shared[256];           // Softmax weights for residual value accumulation

    // 1. Cache the query into shared memory
    for (int d = (int)tid; d < D; d += (int)t_per_tg) {
        q_shared[d] = Q[tg_idx.x * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── PASS 1: Online softmax over ALL tokens (sparse + dense) ──────────────
    SoftmaxState sm_state = { -1e30f, 0.0f };

    // Calculate Split-K partition slice
    const int blocks_per_split = (K + S_split - 1) / S_split;
    const int k_start = (int)tg_idx.y * blocks_per_split;
    const int k_end   = min(k_start + blocks_per_split, K);

    // 1a. Sparse compressed blocks — Project-Then-Attend with anchor-RoPE on VK
    for (int k = k_start; k < k_end; ++k) {
        int slot_id   = slot_indices[k];
        int slen      = seq_lens[slot_id];
        int anchor_pos = anchor_positions[slot_id];

        // Step 1: Rotate anchor key at anchor_pos
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            float raw_ak = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + d);
            if (rope_pool) {
                int   partner   = (d < half_d) ? (d + half_d) : (d - half_d);
                int   idx       = (d < half_d) ? d : (d - half_d);
                float theta     = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                float angle     = anchor_pos * theta;
                float c = cos(angle), s = sin(angle);
                float raw_p = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + partner);
                float rot_c = (d < half_d) ? -raw_p : raw_p;
                ak_rot_shared[d] = raw_ak * c + rot_c * s;
            } else {
                ak_rot_shared[d] = raw_ak;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Anchor dot product (parallel reduction over D)
        float thread_anc = 0.0f;
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            thread_anc += q_shared[d] * ak_rot_shared[d];
        }
        float score_anc = reduce_sum_tg64(thread_anc, red_m, tid);
        if (tid == 0 && k < 64) scores_cached[k * 32] = score_anc * scale;
        
        float block_scale= (float)scales[slot_id];

        if (!approximate_attn) {
            for (int t = (int)tid; t < slen; t += (int)t_per_tg) {
                float scale_u = (float)U_row_scale_pool[slot_id * S_max + t];
                float delta_score = 0.0f;
                int u_off_base = slot_id * S_max * rank + t * rank;
                int pos = anchor_pos + t + 1;

                int ri = -1;
                for (int r = 0; r < max_residual; ++r) {
                    if (res_K_pos[slot_id * max_residual + r] == t) {
                        ri = r;
                        break;
                    }
                }

                for (int d = 0; d < half_d; ++d) {
                    float raw_k1 = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + d);
                    float delta_k1 = 0.0f;
                    int vk_base1 = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                    for (int r = 0; r < rank; ++r) {
                        delta_k1 += (float)U_pool[u_off_base + r] * READ_VAL(VK_pool, vk_base1 + r * n_kv_heads * D);
                    }
                    float k_raw1 = raw_k1 + delta_k1 * scale_u * block_scale;

                    int d2 = d + half_d;
                    float raw_k2 = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + d2);
                    float delta_k2 = 0.0f;
                    int vk_base2 = slot_id * rank * n_kv_heads * D + kv_head * D + d2;
                    for (int r = 0; r < rank; ++r) {
                        delta_k2 += (float)U_pool[u_off_base + r] * READ_VAL(VK_pool, vk_base2 + r * n_kv_heads * D);
                    }
                    float k_raw2 = raw_k2 + delta_k2 * scale_u * block_scale;

                    if (ri != -1) {
                        int base_res = slot_id * max_residual * n_kv_heads * D + ri * n_kv_heads * D + kv_head * D;
                        k_raw1 += READ_VAL(res_K_val, base_res + d);
                        k_raw2 += READ_VAL(res_K_val, base_res + d2);
                    }

                    float k_rot1 = k_raw1;
                    float k_rot2 = k_raw2;
                    if (rope_pool) {
                        float theta = 1.0f / pow(rope_freq_base, (2.0f * d) / D);
                        float angle = pos * theta;
                        float c = cos(angle), s = sin(angle);
                        k_rot1 = k_raw1 * c - k_raw2 * s;
                        k_rot2 = k_raw2 * c + k_raw1 * s;
                    }

                    delta_score += q_shared[d] * k_rot1 + q_shared[d2] * k_rot2;
                }

                float t_score = delta_score * scale;
                if (k < 64 && slen < 31) {
                    scores_cached[k * 32 + 1 + t] = t_score;
                }
                sm_state = merge_softmax_states(sm_state, { t_score, 1.0f });
            }

            if (tid == 0) {
                sm_state = merge_softmax_states(sm_state, { score_anc * scale, 1.0f });
            }
        } else {
            // Step 3: Compute q_proj[r] = q · VK_rot[slot_id, r] (at anchor RoPE)
            if (tid < (uint)rank) {
                float proj_val = 0.0f;
                int base_vk = slot_id * rank * n_kv_heads * D + (int)tid * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float raw_vk = READ_VAL(VK_pool, base_vk + d);
                    float vk_rot;
                    if (rope_pool) {
                        int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        int   idx     = (d < half_d) ? d : (d - half_d);
                        float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                        float angle   = anchor_pos * theta;
                        float c = cos(angle), s = sin(angle);
                        float raw_vkp = READ_VAL(VK_pool, base_vk + partner);
                        float rot_c   = (d < half_d) ? -raw_vkp : raw_vkp;
                        vk_rot = raw_vk * c + rot_c * s;
                    } else {
                        vk_rot = raw_vk;
                    }
                    proj_val += q_shared[d] * vk_rot;
                }
                // ACTIVE_RUNTIME line 222: q_proj = q @ VK * INV_SCALE
                // scale (= 1/sqrt(D)) must be baked into q_proj so that
                // delta_scores = U @ q_proj * block_scale = U @ (q @ VK * scale) * block_scale
                proj_val *= scale;
                q_proj_shared[(int)tid] = proj_val;
                if (k < 64 && slen < 31) scores_cached[k * 32 + 1 + (int)tid] = proj_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                sm_state = merge_softmax_states(sm_state, { score_anc * scale, 1.0f });
            }

            for (int t = (int)tid; t < slen; t += (int)t_per_tg) {
                float delta = 0.0f;
                int u_off = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    delta += q_proj_shared[r] * (float)U_pool[u_off + r];
                }
                // q_proj already includes scale (baked in above).
                // ACTIVE_RUNTIME line 223-224:
                //   delta_scores = U @ q_proj * block_scale  (q_proj has scale baked in)
                //   s = s_anchor + delta_scores
                // s_anchor = score_anc * scale (anchor dot scaled)
                float scale_u = (float)U_row_scale_pool[slot_id * S_max + t];
                float t_score = score_anc * scale + delta * scale_u * block_scale;

                // Add residual score
                int ri = -1;
                for (int r = 0; r < max_residual; ++r) {
                    if (res_K_pos[slot_id * max_residual + r] == t) {
                        ri = r;
                        break;
                    }
                }
                if (ri != -1) {
                    float res_score = 0.0f;
                    int base_res = slot_id * max_residual * n_kv_heads * D + ri * n_kv_heads * D + kv_head * D;
                    int pos = anchor_pos + t + 1;
                    for (int d = 0; d < D; ++d) {
                        float raw_rk = READ_VAL(res_K_val, base_res + d);
                        float rk_rot;
                        if (rope_pool) {
                            int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                            int   idx     = (d < half_d) ? d : (d - half_d);
                            float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                            float angle   = pos * theta;
                            float c = cos(angle), s = sin(angle);
                            float raw_p = READ_VAL(res_K_val, base_res + partner);
                            float rot_c = (d < half_d) ? -raw_p : raw_p;
                            rk_rot = raw_rk * c + rot_c * s;
                        } else {
                            rk_rot = raw_rk;
                        }
                        res_score += q_shared[d] * rk_rot;
                    }
                    t_score += res_score * scale;
                }

                sm_state = merge_softmax_states(sm_state, { t_score, 1.0f });
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 1b. Dense window tokens — exact per-token RoPE
    if (tg_idx.y == 0) {
        for (int t = (int)tid; t < T_dense; t += (int)t_per_tg) {
        int pos    = dense_positions[t];
        int base_k = t * n_kv_heads * D + kv_head * D;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            float raw_k = dense_K[base_k + d];
            float k_rot;
            if (has_rope) {
                int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                int   idx     = (d < half_d) ? d : (d - half_d);
                float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                float angle   = pos * theta;
                float c = cos(angle), s = sin(angle);
                float raw_p = dense_K[base_k + partner];
                float rot_c = (d < half_d) ? -raw_p : raw_p;
                k_rot = raw_k * c + rot_c * s;
            } else {
                k_rot = raw_k;
            }
            score += q_shared[d] * k_rot;
        }
        sm_state = merge_softmax_states(sm_state, { score * scale, 1.0f });
    }
    }
    float global_m = reduce_max_tg64(sm_state.m, red_m, tid);
    float adjusted_d = (sm_state.d == 0.0f) ? 0.0f : (sm_state.d * exp(sm_state.m - global_m));
    float global_d = reduce_sum_tg64(adjusted_d, red_d, tid);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── PASS 2: Accumulate values ─────────────────────────────────────────────
    thread float thread_val[128] = { 0.0f };  // thread-local accumulator [D ≤ 128]
    float norm_factor = (S_split == 1) ? max(global_d, 1e-9f) : 1.0f;

    // 2a. Sparse blocks
    for (int k = k_start; k < k_end; ++k) {
        int slot_id    = slot_indices[k];
        int slen       = seq_lens[slot_id];
        float block_scale = (float)scales[slot_id];

        float score_anc_scaled = 0.0f;
        bool use_cache = (k < 64 && slen < 31);

        if (use_cache) {
            score_anc_scaled = scores_cached[k * 32];
            if (approximate_attn) {
                if (tid < (uint)rank) q_proj_shared[(int)tid] = scores_cached[k * 32 + 1 + (int)tid];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            // k >= 64 or slen >= 31: recompute anchor score
            int anchor_pos = anchor_positions[slot_id];
            for (int d = (int)tid; d < D; d += (int)t_per_tg) {
                float raw_ak = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + d);
                if (rope_pool) {
                    int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    int   idx     = (d < half_d) ? d : (d - half_d);
                    float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                    float angle   = anchor_pos * theta;
                    float c = cos(angle), s = sin(angle);
                    float raw_p = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + partner);
                    float rot_c = (d < half_d) ? -raw_p : raw_p;
                    ak_rot_shared[d] = raw_ak * c + rot_c * s;
                } else {
                    ak_rot_shared[d] = raw_ak;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float thread_anc = 0.0f;
            for (int d = (int)tid; d < D; d += (int)t_per_tg) thread_anc += q_shared[d] * ak_rot_shared[d];
            score_anc_scaled = reduce_sum_tg64(thread_anc, red_m, tid) * scale;

            if (approximate_attn) {
                if (tid < (uint)rank) {
                    float proj_val = 0.0f;
                    int base_vk = slot_id * rank * n_kv_heads * D + (int)tid * n_kv_heads * D + kv_head * D;
                    for (int d = 0; d < D; ++d) {
                        float raw_vk = READ_VAL(VK_pool, base_vk + d);
                        float vk_rot;
                        if (rope_pool) {
                            int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                            int   idx     = (d < half_d) ? d : (d - half_d);
                            float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                            float angle   = anchor_pos * theta;
                            float c = cos(angle), s = sin(angle);
                            float raw_vkp = READ_VAL(VK_pool, base_vk + partner);
                            float rot_c   = (d < half_d) ? -raw_vkp : raw_vkp;
                            vk_rot = raw_vk * c + rot_c * s;
                        } else {
                            vk_rot = raw_vk;
                        }
                        proj_val += q_shared[d] * vk_rot;
                    }
                    proj_val *= scale;  // bake in INV_SCALE — matches ACTIVE_RUNTIME line 222
                    q_proj_shared[(int)tid] = proj_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Anchor weight
        float w_anc = exp(score_anc_scaled - global_m) / norm_factor;

        // Token weights + w_proj accumulation
        float local_sum_w = 0.0f;
        float local_w_proj[64] = { 0.0f };

        for (int t = (int)tid; t < slen; t += (int)t_per_tg) {
            float scale_u = (float)U_row_scale_pool[slot_id * S_max + t];
            float t_score = 0.0f;

            int ri = -1;
            for (int r = 0; r < max_residual; ++r) {
                if (res_K_pos[slot_id * max_residual + r] == t) {
                    ri = r;
                    break;
                }
            }
            float res_score_val = 0.0f;
            if (ri != -1) {
                int base_res = slot_id * max_residual * n_kv_heads * D + ri * n_kv_heads * D + kv_head * D;
                int pos = anchor_positions[slot_id] + t + 1;
                for (int d = 0; d < D; ++d) {
                    float raw_rk = READ_VAL(res_K_val, base_res + d);
                    float rk_rot;
                    if (rope_pool) {
                        int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        int   idx     = (d < half_d) ? d : (d - half_d);
                        float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                        float angle   = pos * theta;
                        float c = cos(angle), s = sin(angle);
                        float raw_p = READ_VAL(res_K_val, base_res + partner);
                        float rot_c = (d < half_d) ? -raw_p : raw_p;
                        rk_rot = raw_rk * c + rot_c * s;
                    } else {
                        rk_rot = raw_rk;
                    }
                    res_score_val += q_shared[d] * rk_rot;
                }
            }

            if (!approximate_attn) {
                if (use_cache) {
                    t_score = scores_cached[k * 32 + 1 + t];
                } else {
                    float delta_score = 0.0f;
                    int u_off_base = slot_id * S_max * rank + t * rank;
                    int pos = anchor_positions[slot_id] + t + 1;

                    for (int d = 0; d < half_d; ++d) {
                        float raw_k1 = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + d);
                        float delta_k1 = 0.0f;
                        int vk_base1 = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                        for (int r = 0; r < rank; ++r) {
                            delta_k1 += (float)U_pool[u_off_base + r] * READ_VAL(VK_pool, vk_base1 + r * n_kv_heads * D);
                        }
                        float k_raw1 = raw_k1 + delta_k1 * scale_u * block_scale;

                        int d2 = d + half_d;
                        float raw_k2 = READ_VAL(anchors_K, slot_id * n_kv_heads * D + kv_head * D + d2);
                        float delta_k2 = 0.0f;
                        int vk_base2 = slot_id * rank * n_kv_heads * D + kv_head * D + d2;
                        for (int r = 0; r < rank; ++r) {
                            delta_k2 += (float)U_pool[u_off_base + r] * READ_VAL(VK_pool, vk_base2 + r * n_kv_heads * D);
                        }
                        float k_raw2 = raw_k2 + delta_k2 * scale_u * block_scale;

                        if (ri != -1) {
                            int base_res = slot_id * max_residual * n_kv_heads * D + ri * n_kv_heads * D + kv_head * D;
                            k_raw1 += READ_VAL(res_K_val, base_res + d);
                            k_raw2 += READ_VAL(res_K_val, base_res + d2);
                        }

                        float k_rot1 = k_raw1;
                        float k_rot2 = k_raw2;
                        if (rope_pool) {
                            float theta = 1.0f / pow(rope_freq_base, (2.0f * d) / D);
                            float angle = pos * theta;
                            float c = cos(angle), s = sin(angle);
                            k_rot1 = k_raw1 * c - k_raw2 * s;
                            k_rot2 = k_raw2 * c + k_raw1 * s;
                        }

                        delta_score += q_shared[d] * k_rot1 + q_shared[d2] * k_rot2;
                    }
                    t_score = delta_score * scale;
                }
            } else {
                float delta = 0.0f;
                int u_off = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) delta += q_proj_shared[r] * (float)U_pool[u_off + r];
                t_score = score_anc_scaled + delta * scale_u * block_scale + res_score_val * scale;
            }

            float w_t = exp(t_score - global_m) / norm_factor;
            local_sum_w += w_t;
            w_shared[t] = w_t;
            int u_off = slot_id * S_max * rank + t * rank;
            for (int r = 0; r < rank; ++r) {
                local_w_proj[r] += w_t * (float)U_pool[u_off + r] * scale_u;
            }
        }

        float total_sum_w = reduce_sum_tg64(local_sum_w, red_m, tid);
        if (tid == 0) {
            red_sum_w = total_sum_w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce w_proj[r]
        for (int r = 0; r < rank; ++r) red_proj_temp[(int)tid * rank + r] = local_w_proj[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < (uint)rank) {
            float s = 0.0f;
            for (int t = 0; t < (int)t_per_tg; ++t) s += red_proj_temp[t * rank + (int)tid];
            red_w_proj[(int)tid] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate anchor + SVD value contributions + residual value contributions.
        float w_total = w_anc + red_sum_w;
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            // Shared anchor base: all tokens (anchor + deltas) contribute through anchors_V
            thread_val[d] += w_total * READ_VAL(anchors_V, slot_id * n_kv_heads * D + kv_head * D + d);
            // SVD delta: weighted sum of U[t] @ VV adds per-token deviations from the anchor base
            float svd_v = 0.0f;
            int base_vv = slot_id * rank * n_kv_heads * D + kv_head * D + d;
            for (int r = 0; r < rank; ++r) {
                svd_v += red_w_proj[r] * READ_VAL(VV_pool, base_vv + r * n_kv_heads * D);
            }
            thread_val[d] += svd_v * block_scale;

            // Add residual value contribution!
            float res_v_sum = 0.0f;
            for (int r = 0; r < max_residual; ++r) {
                int t = res_V_pos[slot_id * max_residual + r];
                if (t >= 0 && t < slen) {
                    int base_res_v = slot_id * max_residual * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    res_v_sum += w_shared[t] * READ_VAL(res_V_val, base_res_v + d);
                }
            }
            thread_val[d] += res_v_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 2b. Dense window tokens
    if (T_dense > 0 && tg_idx.y == 0) {
        const int CHUNK_SIZE = 1024;
        for (int chunk_start = 0; chunk_start < T_dense; chunk_start += CHUNK_SIZE) {
            int chunk_end = min(chunk_start + CHUNK_SIZE, T_dense);
            int chunk_len = chunk_end - chunk_start;

            // Step A: compute dense weight for each token in this chunk
            for (int t = (int)tid; t < chunk_len; t += (int)t_per_tg) {
                int global_t = chunk_start + t;
                int pos    = dense_positions[global_t];
                int base_k = global_t * n_kv_heads * D + kv_head * D;
                float score = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float raw_k = dense_K[base_k + d];
                    float k_rot;
                    if (has_rope) {
                        int   partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        int   idx     = (d < half_d) ? d : (d - half_d);
                        float theta   = 1.0f / pow(rope_freq_base, (2.0f * idx) / D);
                        float angle   = pos * theta;
                        float c = cos(angle), s = sin(angle);
                        float raw_p = dense_K[base_k + partner];
                        float rot_c = (d < half_d) ? -raw_p : raw_p;
                        k_rot = raw_k * c + rot_c * s;
                    } else {
                        k_rot = raw_k;
                    }
                    score += q_shared[d] * k_rot;
                }
                dense_weights[t] = exp(score * scale - global_m) / norm_factor;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step B: Accumulate for the current chunk
            for (int d = (int)tid; d < D; d += (int)t_per_tg) {
                float val_accum = 0.0f;
                for (int t = 0; t < chunk_len; ++t) {
                    int global_t = chunk_start + t;
                    val_accum += dense_weights[t] * dense_V[global_t * n_kv_heads * D + kv_head * D + d];
                }
                thread_val[d] += val_accum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write fully combined (sparse + dense) output
    if (S_split == 1) {
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            out_buf[tg_idx.x * D + d] = thread_val[d];
        }
    } else {
        for (int d = (int)tid; d < D; d += (int)t_per_tg) {
            split_out[tg_idx.x * S_split * D + tg_idx.y * D + d] = thread_val[d];
        }
        if (tid == 0) {
            split_m[tg_idx.x * S_split + tg_idx.y] = global_m;
            split_d[tg_idx.x * S_split + tg_idx.y] = global_d;
        }
    }
}

kernel void merge_split_k_kernel(
    device const float*  split_out    [[buffer(0)]], // [H_q, S_split, D] f32
    device const float*  split_m      [[buffer(1)]], // [H_q, S_split] f32
    device const float*  split_d      [[buffer(2)]], // [H_q, S_split] f32
    device float*        out_buf      [[buffer(3)]], // [H_q, D] f32 final
    device float*        lse_buf      [[buffer(4)]], // [H_q] f32 final
    device const int32_t& S_split     [[buffer(5)]],
    device const int32_t& D           [[buffer(6)]],
    uint tg_idx    [[threadgroup_position_in_grid]],  // query head index 0..H_q-1
    uint tid       [[thread_position_in_threadgroup]],
    uint t_per_tg  [[threads_per_threadgroup]]
) {
    // 1. Find global maximum across splits
    float global_m = -1e30f;
    for (int s = 0; s < S_split; ++s) {
        float m_val = split_m[tg_idx * S_split + s];
        if (m_val > global_m) {
            global_m = m_val;
        }
    }
    
    // 2. Compute global denominator
    float global_d = 0.0f;
    for (int s = 0; s < S_split; ++s) {
        float m_val = split_m[tg_idx * S_split + s];
        float d_val = split_d[tg_idx * S_split + s];
        global_d += d_val * exp(m_val - global_m);
    }
    
    // 3. Merge outputs
    for (int d = (int)tid; d < D; d += (int)t_per_tg) {
        float val_accum = 0.0f;
        for (int s = 0; s < S_split; ++s) {
            float m_val = split_m[tg_idx * S_split + s];
            float local_out = split_out[tg_idx * S_split * D + s * D + d];
            val_accum += local_out * exp(m_val - global_m);
        }
        out_buf[tg_idx * D + d] = val_accum / max(global_d, 1e-9f);
    }
    
    // 4. Write LSE
    if (tid == 0) {
        lse_buf[tg_idx] = global_m + log(max(global_d, 1e-9f));
    }
}
