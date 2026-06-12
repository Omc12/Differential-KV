#include "runtime/diffkv_attention.hpp"
#include "native_core/srl/attention_cache.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace diffkv {

// ── CPU Project-Then-Attend (reference / non-Apple fallback) ──────────────────
// Kept for correctness comparison and CUDA/CPU builds.
// On Apple the Metal kernel in diffkv_attention.mm is used instead.
void execute_cpu_attention(
    const float* Q,
    const int32_t* slots,
    float* cpu_output,
    float* lse_sparse,
    NativeBlockPool* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale,
    bool has_rope, float rope_freq_base, bool approximate_attn
) {
    const float* Q_ptr = Q;
    const int8_t* U = kv_engine->get_host_U();
    const ggml_fp16_t* U_scale = kv_engine->get_host_U_scale();
    const ggml_fp16_t* VK = kv_engine->get_host_VK();
    const ggml_fp16_t* VV = kv_engine->get_host_VV();
    const ggml_fp16_t* anchors_K = kv_engine->get_host_anchors_K();
    const ggml_fp16_t* anchors_V = kv_engine->get_host_anchors_V();
    const int32_t* seq_lens = kv_engine->get_host_seq_lens();
    const ggml_fp16_t* scales = kv_engine->get_host_scales();
    const int32_t* anchor_positions = kv_engine->get_host_anchor_positions();

    // Deduplicate slots
    int n_slots = kv_engine->get_seq_lens()->ne[0];
    std::vector<int32_t> unique_slots;
    unique_slots.reserve(K);
    for (int k = 0; k < K; ++k) {
        int32_t slot_id = slots[k];
        if (slot_id >= 0 && slot_id < n_slots) {
            if (std::find(unique_slots.begin(), unique_slots.end(), slot_id) == unique_slots.end()) {
                unique_slots.push_back(slot_id);
            }
        }
    }
    const int32_t* active_slots = unique_slots.data();
    int active_K = unique_slots.size();

    int half_d = D / 2;
    const int g = n_q_heads / n_kv_heads;

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        float max_score = -1e30f;
        struct SlotScoreInfo {
            float anchor_score;
            std::vector<float> q_proj;
            std::vector<float> token_scores;
        };
        std::vector<SlotScoreInfo> slot_infos(active_K);

        for (int k = 0; k < active_K; ++k) {
            int slot_id = active_slots[k];
            int slen = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            int anchor_pos = anchor_positions[slot_id];

            // 1. Rotated anchor score
            float score_anc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float raw_ak = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                float ak_rot = raw_ak;
                if (has_rope) {
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    float raw_partner = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + partner]);
                    float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                    int idx = (d < half_d) ? d : (d - half_d);
                    float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                    float angle = anchor_pos * theta;
                    ak_rot = raw_ak * std::cos(angle) + rot_contrib * std::sin(angle);
                }
                score_anc += Q_ptr[h * D + d] * ak_rot;
            }
            slot_infos[k].anchor_score = score_anc;
            float s_anc_scaled = score_anc * scale;
            if (s_anc_scaled > max_score) max_score = s_anc_scaled;

            if (!approximate_attn) {
                slot_infos[k].token_scores.resize(slen);
                for (int t = 0; t < slen; ++t) {
                    float score_t = 0.0f;
                    int u_off = slot_id * S_max * rank + t * rank;
                    int pos = anchor_pos + t + 1;
                    
                    for (int d = 0; d < half_d; ++d) {
                        // Reconstruct d
                        float raw_k1 = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                        float delta_k1 = 0.0f;
                        int vk_base1 = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                        for (int r = 0; r < rank; ++r) {
                            delta_k1 += (float)U[u_off + r] * ggml_fp16_to_fp32(VK[vk_base1 + r * n_kv_heads * D]);
                        }
                        float k_raw1 = raw_k1 + delta_k1 * scale_u * block_scale;

                        // Reconstruct d2
                        int d2 = d + half_d;
                        float raw_k2 = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d2]);
                        float delta_k2 = 0.0f;
                        int vk_base2 = slot_id * rank * n_kv_heads * D + kv_head * D + d2;
                        for (int r = 0; r < rank; ++r) {
                            delta_k2 += (float)U[u_off + r] * ggml_fp16_to_fp32(VK[vk_base2 + r * n_kv_heads * D]);
                        }
                        float k_raw2 = raw_k2 + delta_k2 * scale_u * block_scale;

                        // Rotate
                        float k_rot1 = k_raw1;
                        float k_rot2 = k_raw2;
                        if (has_rope) {
                            float theta = 1.0f / std::pow(rope_freq_base, (2.0f * d) / D);
                            float angle = pos * theta;
                            float cos_a = std::cos(angle);
                            float sin_a = std::sin(angle);
                            k_rot1 = k_raw1 * cos_a - k_raw2 * sin_a;
                            k_rot2 = k_raw2 * cos_a + k_raw1 * sin_a;
                        }
                        score_t += Q_ptr[h * D + d] * k_rot1 + Q_ptr[h * D + d2] * k_rot2;
                    }
                    float t_score = score_t * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;
                }
            } else {
                // 2. Project-Then-Attend: q_proj[r] = q · VK_rot[r] at anchor pos
                slot_infos[k].q_proj.resize(rank, 0.0f);
                for (int r = 0; r < rank; ++r) {
                    float proj = 0.0f;
                    int base_vk = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                    for (int d = 0; d < D; ++d) {
                        float raw_vk = ggml_fp16_to_fp32(VK[base_vk + d]);
                        float vk_rot = raw_vk;
                        if (has_rope) {
                            int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                            float raw_vkp = ggml_fp16_to_fp32(VK[base_vk + partner]);
                            float rot_c = (d < half_d) ? -raw_vkp : raw_vkp;
                            int idx = (d < half_d) ? d : (d - half_d);
                            float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                            float angle = anchor_pos * theta;
                            vk_rot = raw_vk * std::cos(angle) + rot_c * std::sin(angle);
                        }
                        proj += Q_ptr[h * D + d] * vk_rot;
                    }
                    slot_infos[k].q_proj[r] = proj;
                }

                // 3. Token delta scores
                slot_infos[k].token_scores.resize(slen);
                for (int t = 0; t < slen; ++t) {
                    float delta = 0.0f;
                    int u_off = slot_id * S_max * rank + t * rank;
                    for (int r = 0; r < rank; ++r) {
                        delta += slot_infos[k].q_proj[r] * (float)U[u_off + r];
                    }
                    float t_score = (delta * scale_u * block_scale + score_anc) * scale;
                    slot_infos[k].token_scores[t] = t_score;
                    if (t_score > max_score) max_score = t_score;
                }
            }
        }

        // Softmax denominator
        double sum_exp = 0.0;
        for (int k = 0; k < active_K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores) {
                sum_exp += std::exp(s - max_score);
            }
        }
        lse_sparse[h] = max_score + std::log(std::max(sum_exp, 1e-9));

        // Value accumulation
        std::vector<double> accum(D, 0.0);
        for (int k = 0; k < active_K; ++k) {
            int slot_id = active_slots[k];
            int slen = seq_lens[slot_id];
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w = 0.0;
            std::vector<double> w_proj(rank, 0.0);

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w += w_t;
                int u_off = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    w_proj[r] += w_t * (float)U[u_off + r] * scale_u;
                }
            }

            double w_total = w_anc + sum_w;
            for (int d = 0; d < D; ++d) {
                accum[d] += w_total * ggml_fp16_to_fp32(anchors_V[slot_id * n_kv_heads * D + kv_head * D + d]);
                double svd_v = 0.0;
                int base_vv = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                for (int r = 0; r < rank; ++r) {
                    svd_v += w_proj[r] * ggml_fp16_to_fp32(VV[base_vv + r * n_kv_heads * D]);
                }
                accum[d] += svd_v * block_scale;
            }
        }
        for (int d = 0; d < D; ++d) cpu_output[h * D + d] = (float)accum[d];
    }
}

// ── CPU dense window attention helper (used by CPU fallback path only) ────────
static void cpu_dense_attention(
    const float* Q_ptr,          // [n_q_heads, D]
    const float* active_k_dense, // [T, n_kv, D]
    const float* active_v_dense, // [T, n_kv, D]
    const int32_t* positions,    // [T] actual sequence positions (or nullptr)
    int T, int n_q_heads, int n_kv_heads, int D,
    float scale, bool has_rope, float rope_freq_base,
    int anchor_pos,              // fallback start position if positions == nullptr
    float* out_dense,            // [n_q_heads, D] — written
    float* lse_dense_out         // [n_q_heads]   — written
) {
    const int g = n_q_heads / n_kv_heads;
    const int half_d = D / 2;

    std::vector<float> theta(half_d);
    for (int idx = 0; idx < half_d; ++idx) {
        theta[idx] = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
    }

    // Pre-rotate K for each KV head
    std::vector<float> K_rot(T * n_kv_heads * D);
    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        for (int t = 0; t < T; ++t) {
            int pos = positions ? positions[t] : (anchor_pos + t);
            int src = t * n_kv_heads * D + kv_head * D;
            int dst_off = (t * n_kv_heads + kv_head) * D;
            for (int d = 0; d < D; ++d) {
                float raw_k = active_k_dense[src + d];
                if (has_rope) {
                    int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                    float raw_p = active_k_dense[src + partner];
                    float rot_c = (d < half_d) ? -raw_p : raw_p;
                    int idx = (d < half_d) ? d : (d - half_d);
                    float angle = pos * theta[idx];
                    K_rot[dst_off + d] = raw_k * std::cos(angle) + rot_c * std::sin(angle);
                } else {
                    K_rot[dst_off + d] = raw_k;
                }
            }
        }
    }

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;
        const float* Kh = K_rot.data() + kv_head * T * D; // not right index

        float max_s = -1e30f;
        std::vector<float> scores(T);
        for (int t = 0; t < T; ++t) {
            float dot = 0.0f;
            // K_rot layout: [T, n_kv, D] — stride per kv_head
            const float* k_t = K_rot.data() + (t * n_kv_heads + kv_head) * D;
            for (int d = 0; d < D; ++d) dot += Q_ptr[h * D + d] * k_t[d];
            scores[t] = dot * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }
        float sum_e = 0.0f;
        for (int t = 0; t < T; ++t) sum_e += std::exp(scores[t] - max_s);
        lse_dense_out[h] = max_s + std::log(std::max(sum_e, 1e-9f));

        for (int d = 0; d < D; ++d) out_dense[h * D + d] = 0.0f;
        for (int t = 0; t < T; ++t) {
            float w = std::exp(scores[t] - lse_dense_out[h]);
            const float* v_t = active_v_dense + (t * n_kv_heads + kv_head) * D;
            for (int d = 0; d < D; ++d) out_dense[h * D + d] += w * v_t[d];
        }
    }
}

// ── Main GGML custom-op callback ──────────────────────────────────────────────
void custom_attention_op_callback(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,  // Q  [n_q_heads, D]
    const struct ggml_tensor * b,  // slot_indices [K]
    const struct ggml_tensor * c,  // kv_concat of current token [2, n_kv, D]
    int ith, int nth,
    void * userdata
) {
    if (ith != 0) return;

    const struct ggml_tensor* Q             = a;
    const struct ggml_tensor* slot_indices  = b;
    CustomAttnUserData* data = static_cast<CustomAttnUserData*>(userdata);

    const int n_q_heads = data->n_q_heads;
    const int n_kv_heads = data->n_kv_heads;
    const int D = data->D;
    const int F_kv = n_kv_heads * D;

    bool all_reused = false;
    bool cache_active = (get_global_attn_cache().threshold <= 1.0f);
    std::vector<float> q_host;
    if (cache_active) {
        q_host.resize(n_q_heads * D);
        ggml_backend_tensor_get(Q, q_host.data(), 0, n_q_heads * D * sizeof(float));

        std::vector<bool> reuse_mask(n_q_heads, false);
        std::vector<float> out_cached(n_q_heads * D, 0.0f);
        bool has_cache = get_global_attn_cache().check_and_update(
            data->session_id,
            data->layer_idx,
            q_host.data(),
            n_q_heads,
            D,
            reuse_mask,
            out_cached
        );
        all_reused = has_cache && std::all_of(reuse_mask.begin(), reuse_mask.end(), [](bool b) { return b; });
        if (all_reused) {
            std::memcpy(dst->data, out_cached.data(), n_q_heads * D * sizeof(float));
            return;
        }
    }

    // ── Step 1: Append current token K/V to the dense buffer ─────────────────
    // The dense buffer (active_k_dense / active_v_dense) is a flat CPU array
    // maintained by main.cpp. When ignore_c == false, we append the current
    // token's projected K/V here so the Metal kernel can see it.
    int T_dense = data->active_block_tokens;  // count of tokens already in dense buf

    if (c != nullptr && !data->ignore_c &&
        data->active_k_dense != nullptr && data->active_v_dense != nullptr) {

        // Read current token K/V from GGML tensor c = ggml_concat(k, v)
        std::vector<float> cur_kv(2 * F_kv, 0.0f);
        if (c->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(2 * F_kv);
            ggml_backend_tensor_get(c, tmp.data(), 0, 2 * F_kv * sizeof(ggml_fp16_t));
            for (int i = 0; i < 2 * F_kv; ++i) cur_kv[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(c, cur_kv.data(), 0, 2 * F_kv * sizeof(float));
        }

        // Append to dense buffer at slot T_dense
        float* k_dst = data->active_k_dense + (size_t)T_dense * F_kv;
        float* v_dst = data->active_v_dense + (size_t)T_dense * F_kv;
        std::memcpy(k_dst, cur_kv.data(),         F_kv * sizeof(float));
        std::memcpy(v_dst, cur_kv.data() + F_kv,  F_kv * sizeof(float));

        // Record the actual sequence position for this token so the kernel can
        // apply exact per-token RoPE.  current_pos is set by main.cpp each step.
        if (data->active_positions_dense != nullptr) {
            data->active_positions_dense[T_dense] = data->current_pos;
        }
        T_dense++;
    }

    // ── Step 2: Dispatch GPU or CPU attention ─────────────────────────────────
#ifdef __APPLE__
    const char* env_cpu = std::getenv("DIFFKV_FORCE_CPU_ATTN");
    bool force_cpu = (env_cpu && std::string(env_cpu) == "1");

    if (!force_cpu) {
        // Unified Metal path: one kernel dispatch handles sparse + dense.
        // Output is written directly into dst->data.
        std::vector<float> lse_dummy(n_q_heads, -1e30f);
        execute_metal_attention(
            dst, Q, (struct ggml_tensor*)slot_indices, data,
            lse_dummy.data(),
            data->active_k_dense,
            data->active_v_dense,
            data->active_positions_dense,
            T_dense
        );


        if (cache_active) {
            get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
        }
        return;
    }
#endif

    // ── CPU fallback (non-Apple builds or DIFFKV_FORCE_CPU_ATTN=1) ───────────
    const int K = data->K;
    const int rank = data->rank;
    const int S_max = data->S_max;
    const float scale = data->scale;
    const bool has_rope = data->has_rope;
    const float rope_freq_base = data->rope_freq_base;
    NativeBlockPool* kv_engine = data->kv_engine;

    // Sparse attention (CPU Project-Then-Attend)
    std::vector<float> out_sparse(n_q_heads * D, 0.0f);
    std::vector<float> lse_sparse(n_q_heads, -1e30f);
    if (K > 0 && slot_indices && slot_indices->data) {
        execute_cpu_attention(
            (const float*)Q->data,
            (const int32_t*)slot_indices->data,
            out_sparse.data(), lse_sparse.data(),
            kv_engine,
            n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
            has_rope, rope_freq_base, data->approximate_attn
        );
    }

    if (T_dense == 0) {
        // No dense tokens: copy sparse output (or zero)
        if (K > 0) {
            std::memcpy(dst->data, out_sparse.data(), n_q_heads * D * sizeof(float));
        } else {
            std::memset(dst->data, 0, n_q_heads * D * sizeof(float));
        }
        if (cache_active) {
            get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
        }
        return;
    }

    // Dense window attention
    std::vector<float> out_dense(n_q_heads * D, 0.0f);
    std::vector<float> lse_dense(n_q_heads, -1e30f);
    cpu_dense_attention(
        (const float*)Q->data,
        data->active_k_dense, data->active_v_dense,
        data->active_positions_dense,
        T_dense, n_q_heads, n_kv_heads, D,
        scale, has_rope, rope_freq_base,
        data->active_slot,
        out_dense.data(), lse_dense.data()
    );

    // LSE combine
    std::vector<float> final_out(n_q_heads * D);
    for (int h = 0; h < n_q_heads; ++h) {
        float ld = lse_dense[h];
        float ls = lse_sparse[h];
        if (K == 0) {
            // Only dense
            for (int d = 0; d < D; ++d) final_out[h * D + d] = out_dense[h * D + d];
        } else {
            float lse_max = std::max(ld, ls);
            float wd = std::exp(ld - lse_max);
            float ws = std::exp(ls - lse_max);
            float denom = std::max(wd + ws, 1e-9f);
            for (int d = 0; d < D; ++d) {
                final_out[h * D + d] = (out_dense[h * D + d] * wd + out_sparse[h * D + d] * ws) / denom;
            }
        }
    }
    std::memcpy(dst->data, final_out.data(), n_q_heads * D * sizeof(float));
    if (cache_active) {
        get_global_attn_cache().save(data->session_id, data->layer_idx, q_host.data(), n_q_heads, D, (const float*)dst->data);
    }
}

} // namespace diffkv
