#include "runtime/diffkv_attention.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace diffkv {

// Standard CPU Project-Then-Attend reference attention implementation
void execute_cpu_attention(
    const float* Q,
    const int32_t* slots,
    float* cpu_output,
    float* lse_sparse,
    NativeBlockPool* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale,
    bool has_rope, float rope_freq_base
) {
    const float* Q_ptr = Q;
    const int8_t* U = (const int8_t*)kv_engine->get_U()->data;
    const ggml_fp16_t* U_scale = (const ggml_fp16_t*)kv_engine->get_U_scale()->data;
    const ggml_fp16_t* VK = (const ggml_fp16_t*)kv_engine->get_VK()->data;
    const ggml_fp16_t* VV = (const ggml_fp16_t*)kv_engine->get_VV()->data;
    const ggml_fp16_t* anchors_K = (const ggml_fp16_t*)kv_engine->get_anchors_K()->data;
    const ggml_fp16_t* anchors_V = (const ggml_fp16_t*)kv_engine->get_anchors_V()->data;
    const int32_t* seq_lens = (const int32_t*)kv_engine->get_seq_lens()->data;
    const ggml_fp16_t* scales = (const ggml_fp16_t*)kv_engine->get_scales()->data;
    const int32_t* anchor_positions = (const int32_t*)kv_engine->get_anchor_positions()->data;

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
        std::vector<SlotScoreInfo> slot_infos(K);

        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            int anchor_pos = anchor_positions[slot_id];

            // 1. Anchor score (rotated by RoPE)
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

            // 2. Query projection (using rotated VK)
            std::vector<float> q_proj(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                float proj_val = 0.0f;
                int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float raw_vk = ggml_fp16_to_fp32(VK[base_vk_offset + d]);
                    float vk_rot = raw_vk;
                    if (has_rope) {
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_partner = ggml_fp16_to_fp32(VK[base_vk_offset + partner]);
                        float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                        int idx = (d < half_d) ? d : (d - half_d);
                        float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                        float angle = anchor_pos * theta;
                        vk_rot = raw_vk * std::cos(angle) + rot_contrib * std::sin(angle);
                    }
                    proj_val += Q_ptr[h * D + d] * vk_rot;
                }
                q_proj[r] = proj_val;
            }
            slot_infos[k].q_proj = q_proj;

            // 3. Delta scores
            slot_infos[k].token_scores.resize(slen);
            for (int t = 0; t < slen; ++t) {
                float delta_sum = 0.0f;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    delta_sum += q_proj[r] * static_cast<float>(U[u_offset + r]);
                }
                float t_score = (delta_sum * scale_u * block_scale + score_anc) * scale;
                slot_infos[k].token_scores[t] = t_score;
                if (t_score > max_score) max_score = t_score;
            }
        }

        // Softmax denominator
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores) {
                sum_exp += std::exp(s - max_score);
            }
        }
        lse_sparse[h] = max_score + std::log(std::max(sum_exp, 1e-9));

        // Pass 2: Accumulate values
        std::vector<double> accum_val(D, 0.0);
        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);

            double w_anc = std::exp(slot_infos[k].anchor_score * scale - max_score) / sum_exp;
            double sum_w_tokens = 0.0;
            std::vector<double> w_proj(rank, 0.0);

            for (int t = 0; t < slen; ++t) {
                double w_t = std::exp(slot_infos[k].token_scores[t] - max_score) / sum_exp;
                sum_w_tokens += w_t;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int r = 0; r < rank; ++r) {
                    w_proj[r] += w_t * static_cast<float>(U[u_offset + r]) * scale_u;
                }
            }

            double w_total_anc = w_anc + sum_w_tokens;

            for (int d = 0; d < D; ++d) {
                double av_val = ggml_fp16_to_fp32(anchors_V[slot_id * n_kv_heads * D + kv_head * D + d]);
                accum_val[d] += w_total_anc * av_val;

                double svd_v_contrib = 0.0;
                int base_vv_offset = slot_id * rank * n_kv_heads * D + kv_head * D + d;
                for (int r = 0; r < rank; ++r) {
                    double vv_val = ggml_fp16_to_fp32(VV[base_vv_offset + r * n_kv_heads * D]);
                    svd_v_contrib += w_proj[r] * vv_val;
                }
                accum_val[d] += svd_v_contrib * block_scale;
            }
        }

        for (int d = 0; d < D; ++d) {
            cpu_output[h * D + d] = static_cast<float>(accum_val[d]);
        }
    }
}

void custom_attention_op_callback(
    struct ggml_tensor * dst,
    const struct ggml_tensor * a,
    const struct ggml_tensor * b,
    const struct ggml_tensor * c,
    int ith,
    int nth,
    void * userdata
) {
    if (ith != 0) return;

    const struct ggml_tensor * Q = a;
    const struct ggml_tensor * slot_indices = b;

    CustomAttnUserData * data = static_cast<CustomAttnUserData*>(userdata);
    int n_q_heads = data->n_q_heads;
    int n_kv_heads = data->n_kv_heads;
    int rank = data->rank;
    int S_max = data->S_max;
    int K = data->K;
    int D = data->D;
    float scale = data->scale;
    bool has_rope = data->has_rope;
    float rope_freq_base = data->rope_freq_base;
    NativeBlockPool* kv_engine = data->kv_engine;


    std::vector<float> out_sparse(n_q_heads * D, 0.0f);
    std::vector<float> lse_sparse(n_q_heads, -1e30f);

#ifdef __APPLE__
    // On macOS, run GPU Metal sparse attention
    if (K > 0) {
        execute_metal_attention(dst, Q, (struct ggml_tensor*)slot_indices, data, lse_sparse.data());
        
        // Copy sparse output back for blending on host
        memcpy(out_sparse.data(), dst->data, n_q_heads * D * sizeof(float));
    }
#else
    // On non-macOS (CUDA/CPU fallback), run CPU Project-Then-Attend attention
    if (K > 0) {
        execute_cpu_attention(
            (const float*)Q->data,
            (const int32_t*)slot_indices->data,
            out_sparse.data(),
            lse_sparse.data(),
            kv_engine,
            n_q_heads, n_kv_heads, rank, S_max, K, D, scale,
            has_rope, rope_freq_base
        );
    }
#endif

    // ── Dense Window Attention & LSE Combine on CPU ──
    bool has_dense = (data->active_block_tokens > 0 || c != nullptr) && data->active_k_dense != nullptr && data->active_v_dense != nullptr;
    if (has_dense) {
        int active_block_tokens = data->active_block_tokens;
        int anchor_pos = data->active_slot; // We passed start position as active_slot
        const float* active_k_dense = data->active_k_dense;
        const float* active_v_dense = data->active_v_dense;
        const int g = n_q_heads / n_kv_heads;
        const int half_d = D / 2;

        // Append current token's key/value if available
        int F_test = n_kv_heads * D;
        std::vector<float> cur_k(F_test);
        std::vector<float> cur_v(F_test);
        if (c) {
            if (c->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> cur_kv_f16(2 * F_test);
                ggml_backend_tensor_get(c, cur_kv_f16.data(), 0, 2 * F_test * sizeof(ggml_fp16_t));
                for (int i = 0; i < F_test; ++i) {
                    cur_k[i] = ggml_fp16_to_fp32(cur_kv_f16[i]);
                    cur_v[i] = ggml_fp16_to_fp32(cur_kv_f16[F_test + i]);
                }
            } else {
                std::vector<float> cur_kv_f32(2 * F_test);
                ggml_backend_tensor_get(c, cur_kv_f32.data(), 0, 2 * F_test * sizeof(float));
                for (int i = 0; i < F_test; ++i) {
                    cur_k[i] = cur_kv_f32[i];
                    cur_v[i] = cur_kv_f32[F_test + i];
                }
            }

            float* active_k_ptr = const_cast<float*>(active_k_dense);
            float* active_v_ptr = const_cast<float*>(active_v_dense);
            int offset = active_block_tokens * F_test;
            for (int i = 0; i < F_test; ++i) {
                active_k_ptr[offset + i] = cur_k[i];
                active_v_ptr[offset + i] = cur_v[i];
            }

            active_block_tokens++;
        }

        std::vector<float> Q_fp32(n_q_heads * D);
        if (Q->type == GGML_TYPE_F16) {
            const ggml_fp16_t* Q_fp16 = (const ggml_fp16_t*)Q->data;
            for (int i = 0; i < n_q_heads * D; ++i) {
                Q_fp32[i] = ggml_fp16_to_fp32(Q_fp16[i]);
            }
        } else {
            memcpy(Q_fp32.data(), Q->data, n_q_heads * D * sizeof(float));
        }
        const float* Q_ptr = Q_fp32.data();
        std::vector<float> final_output(n_q_heads * D, 0.0f);


        for (int h = 0; h < n_q_heads; ++h) {
            int kv_head = h / g;

            // Reconstruct K and V for the active block dense tokens
            std::vector<float> K_dense_rot(active_block_tokens * D);
            std::vector<float> V_dense(active_block_tokens * D);

            for (int t = 0; t < active_block_tokens; ++t) {
                int pos = anchor_pos + t;

                for (int d = 0; d < D; ++d) {
                    int offset = t * n_kv_heads * D + kv_head * D;
                    float raw_k = active_k_dense[offset + d];
                    float raw_v = active_v_dense[offset + d];
                    V_dense[t * D + d] = raw_v;

                    if (has_rope) {
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_partner = active_k_dense[offset + partner];
                        float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                        int idx = (d < half_d) ? d : (d - half_d);
                        float theta = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
                        float angle = pos * theta;
                        K_dense_rot[t * D + d] = raw_k * std::cos(angle) + rot_contrib * std::sin(angle);
                    } else {
                        K_dense_rot[t * D + d] = raw_k;
                    }
                }
            }

            // Compute query-key dot products
            std::vector<float> scores(active_block_tokens);
            float max_score = -1e30f;
            for (int t = 0; t < active_block_tokens; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += Q_ptr[h * D + d] * K_dense_rot[t * D + d];
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            // Log-Sum-Exp
            float sum_exp = 0.0f;
            for (int t = 0; t < active_block_tokens; ++t) {
                sum_exp += std::exp(scores[t] - max_score);
            }
            float lse_dense = max_score + std::log(std::max(sum_exp, 1e-9f));

            // Compute dense attention output vector
            std::vector<float> out_dense(D, 0.0f);
            for (int t = 0; t < active_block_tokens; ++t) {
                float w_t = std::exp(scores[t] - lse_dense);
                for (int d = 0; d < D; ++d) {
                    out_dense[d] += w_t * V_dense[t * D + d];
                }
            }



            // Combine with sparse attention if sparse blocks exist
            if (K > 0) {
                float lse_sparse_val = lse_sparse[h];
                float lse_max = std::max(lse_dense, lse_sparse_val);
                float w_dense = std::exp(lse_dense - lse_max);
                float w_sparse = std::exp(lse_sparse_val - lse_max);
                float denom = w_dense + w_sparse;

                for (int d = 0; d < D; ++d) {
                    final_output[h * D + d] = (out_dense[d] * w_dense + out_sparse[h * D + d] * w_sparse) / std::max(denom, 1e-9f);
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    final_output[h * D + d] = out_dense[d];
                }
            }
        }
        memcpy(dst->data, final_output.data(), n_q_heads * D * sizeof(float));
    } else {
        // No dense tokens, copy sparse output directly
#ifndef __APPLE__
        if (K > 0) {
            memcpy(dst->data, out_sparse.data(), n_q_heads * D * sizeof(float));
        } else {
            memset(dst->data, 0, n_q_heads * D * sizeof(float));
        }
#endif
    }
}

} // namespace diffkv
