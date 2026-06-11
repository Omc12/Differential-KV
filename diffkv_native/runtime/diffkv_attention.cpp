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

    // Deduplicate slots to prevent token double-processing and softmax distortion
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

    static int call_count = 0;
    bool should_print = (call_count == 0);
    if (should_print) {
        call_count++;
        std::cerr << "\n[C++ CPU ATTN DEBUG - Layer 0 Step 0]\n";
        std::cerr << "  D: " << D << ", rank: " << rank << ", K: " << active_K << " (original K: " << K << "), scale: " << scale << ", has_rope: " << has_rope << ", rope_freq_base: " << rope_freq_base << "\n";
        std::cerr << "  Q[head 0] (first 10):";
        for (int d = 0; d < 10; ++d) std::cerr << " " << Q_ptr[d];
        std::cerr << "\n  slots (all active K):";
        for (int k = 0; k < active_K; ++k) {
            int slot_id = active_slots[k];
            std::cerr << " " << slot_id << "(pos=" << anchor_positions[slot_id] << ")";
        }
        std::cerr << "\n";
    }

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

            if (should_print && h == 0 && k == 0) {
                std::cerr << "  [Block 0, Head 0]\n";
                std::cerr << "    slot_id: " << slot_id << ", slen: " << slen << ", scale_u: " << scale_u << ", block_scale: " << block_scale << ", anchor_pos: " << anchor_pos << "\n";
                std::cerr << "    raw_ak (first 10):";
                for (int d = 0; d < 10; ++d) {
                    std::cerr << " " << ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                }
                std::cerr << "\n";
            }

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

            if (should_print && h == 0 && k == 0) {
                std::cerr << "    score_anc: " << score_anc << "\n";
            }

            float s_anc_scaled = score_anc * scale;
            if (s_anc_scaled > max_score) max_score = s_anc_scaled;

            // 2. Exact reconstruction + per-token RoPE
            slot_infos[k].token_scores.resize(slen);
            for (int t = 0; t < slen; ++t) {
                float t_score_sum = 0.0f;
                int u_offset = slot_id * S_max * rank + t * rank;
                for (int d = 0; d < half_d; ++d) {
                    // Reconstruct partner 1 (d)
                    float raw_ak_1 = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                    float sum_r_vk_1 = 0.0f;
                    for (int r = 0; r < rank; ++r) {
                        int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                        float raw_vk = ggml_fp16_to_fp32(VK[base_vk_offset + d]);
                        float dequant_u = static_cast<float>(U[u_offset + r]) * scale_u;
                        sum_r_vk_1 += dequant_u * raw_vk;
                    }
                    float k_unrot_1 = raw_ak_1 + block_scale * sum_r_vk_1;

                    // Reconstruct partner 2 (d + half_d)
                    int d2 = d + half_d;
                    float raw_ak_2 = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d2]);
                    float sum_r_vk_2 = 0.0f;
                    for (int r = 0; r < rank; ++r) {
                        int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                        float raw_vk = ggml_fp16_to_fp32(VK[base_vk_offset + d2]);
                        float dequant_u = static_cast<float>(U[u_offset + r]) * scale_u;
                        sum_r_vk_2 += dequant_u * raw_vk;
                    }
                    float k_unrot_2 = raw_ak_2 + block_scale * sum_r_vk_2;

                    float k_rot_1 = k_unrot_1;
                    float k_rot_2 = k_unrot_2;
                    if (has_rope) {
                        float theta = 1.0f / std::pow(rope_freq_base, (2.0f * d) / D);
                        int pos = anchor_pos + 1 + t;
                        float angle = pos * theta;
                        k_rot_1 = k_unrot_1 * std::cos(angle) - k_unrot_2 * std::sin(angle);
                        k_rot_2 = k_unrot_2 * std::cos(angle) + k_unrot_1 * std::sin(angle);
                    }
                    t_score_sum += Q_ptr[h * D + d] * k_rot_1 + Q_ptr[h * D + d2] * k_rot_2;
                }
                float t_score = t_score_sum * scale;
                slot_infos[k].token_scores[t] = t_score;
                if (t_score > max_score) max_score = t_score;
            }

            if (should_print && h == 0 && k == 0) {
                std::cerr << "    token_scores (first 10):";
                for (int t = 0; t < std::min(10, slen); ++t) std::cerr << " " << slot_infos[k].token_scores[t];
                std::cerr << "\n";
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

        // Pass 2: Accumulate values
        std::vector<double> accum_val(D, 0.0);
        for (int k = 0; k < active_K; ++k) {
            int slot_id = active_slots[k];
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

        if (should_print && h == 0) {
            std::cerr << "    lse_sparse[0]: " << lse_sparse[0] << "\n";
            std::cerr << "    out_sparse[0] (first 10):";
            for (int d = 0; d < 10; ++d) std::cerr << " " << cpu_output[d];
            std::cerr << "\n";
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
    // On macOS, run GPU Metal sparse attention unless forced to CPU
    bool force_cpu_attn = false;
    if (const char* env_force = std::getenv("DIFFKV_FORCE_CPU_ATTN")) {
        if (std::string(env_force) == "1") {
            force_cpu_attn = true;
        }
    }
    if (K > 0) {
        if (!force_cpu_attn) {
            execute_metal_attention(dst, Q, (struct ggml_tensor*)slot_indices, data, lse_sparse.data());
            // Copy sparse output back for blending on host
            memcpy(out_sparse.data(), dst->data, n_q_heads * D * sizeof(float));
        } else {
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
    bool has_dense = (data->active_block_tokens > 0 || (c != nullptr && !data->ignore_c)) && data->active_k_dense != nullptr && data->active_v_dense != nullptr;
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
        if (c && !data->ignore_c) {
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

        // Precompute theta values for RoPE
        std::vector<float> theta(half_d);
        for (int idx = 0; idx < half_d; ++idx) {
            theta[idx] = 1.0f / std::pow(rope_freq_base, (2.0f * idx) / D);
        }

        std::vector<std::vector<float>> K_dense_rot_by_kv(n_kv_heads, std::vector<float>(active_block_tokens * D));
        std::vector<std::vector<float>> V_dense_by_kv(n_kv_heads, std::vector<float>(active_block_tokens * D));

        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            float* K_rot = K_dense_rot_by_kv[kv_head].data();
            float* V_d = V_dense_by_kv[kv_head].data();
            for (int t = 0; t < active_block_tokens; ++t) {
                int pos = (data->active_positions_dense != nullptr) ? data->active_positions_dense[t] : (anchor_pos + t);
                int t_offset = t * D;
                int src_offset = t * n_kv_heads * D + kv_head * D;

                for (int d = 0; d < D; ++d) {
                    float raw_k = active_k_dense[src_offset + d];
                    float raw_v = active_v_dense[src_offset + d];
                    V_d[t_offset + d] = raw_v;

                    if (has_rope) {
                        int partner = (d < half_d) ? (d + half_d) : (d - half_d);
                        float raw_partner = active_k_dense[src_offset + partner];
                        float rot_contrib = (d < half_d) ? -raw_partner : raw_partner;
                        int idx = (d < half_d) ? d : (d - half_d);
                        float angle = pos * theta[idx];
                        K_rot[t_offset + d] = raw_k * std::cos(angle) + rot_contrib * std::sin(angle);
                    } else {
                        K_rot[t_offset + d] = raw_k;
                    }
                }
            }
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
            const float* K_dense_rot = K_dense_rot_by_kv[kv_head].data();
            const float* V_dense = V_dense_by_kv[kv_head].data();

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
