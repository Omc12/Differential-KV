#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "runtime/diffkv_model.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../third_party/llama.cpp/ggml/src/ggml-impl.h"
#include "runtime/native_block_pool.hpp"
#include "native_core/srl/diffkv_srl.hpp"
#include "runtime/diffkv_attention.hpp"
#include "native_core/compression/async_compressor.hpp"
#include "native_core/kv_runtime_manager.hpp"

using namespace diffkv;

struct ggml_backend_owner {
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ggml_backend_owner() {
        gpu_backend = ggml_backend_init_best();
        cpu_backend = ggml_backend_cpu_init();
        if (!gpu_backend) {
            gpu_backend = cpu_backend;
        }
        ggml_backend_t backends[] = { gpu_backend, cpu_backend };
        sched = ggml_backend_sched_new(backends, NULL, 2, 8192, false, true);
    }

    ~ggml_backend_owner() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (gpu_backend && gpu_backend != cpu_backend) {
            ggml_backend_free(gpu_backend);
        }
        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
    }
};

// Helper to build the Qwen 2.5 dense prefill graph using causal flash attention
struct ggml_cgraph * build_prefill_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_tokens,
    struct ggml_tensor * positions,
    struct ggml_tensor * mask,
    struct ggml_tensor ** out_logits,
    std::vector<struct ggml_tensor *>* k_layers = nullptr,
    std::vector<struct ggml_tensor *>* v_layers = nullptr,
    bool need_logits = false
) {
    const auto & config = model.get_config();
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // 1. Embedding lookup
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.get_token_embd(), input_tokens);

    for (int l = 0; l < config.n_layer; ++l) {
        const auto & layer = model.get_layers()[l];

        // 2. Attention RMSNorm
        struct ggml_tensor * h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.attn_norm);

        // 3. QKV Projections
        struct ggml_tensor * q = ggml_mul_mat(ctx, layer.wq, h);
        if (layer.bq) q = ggml_add(ctx, q, layer.bq);
        struct ggml_tensor * k = ggml_mul_mat(ctx, layer.wk, h);
        if (layer.bk) k = ggml_add(ctx, k, layer.bk);
        struct ggml_tensor * v = ggml_mul_mat(ctx, layer.wv, h);
        if (layer.bv) v = ggml_add(ctx, v, layer.bv);

        // Export raw key/value tensors of each layer (before RoPE)
        if (k_layers) (*k_layers)[l] = k;
        if (v_layers) (*v_layers)[l] = v;

        // 4. RoPE
        int head_dim = config.n_embd / config.n_head;
        struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim, config.n_head, q->ne[1]);
        struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, head_dim, config.n_head_kv, k->ne[1]);

        struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, positions, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * k_rope = ggml_rope_ext(ctx, k_reshaped, positions, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // 5. Permute for Flash Attention
        struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        struct ggml_tensor * k_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim, config.n_head_kv, v->ne[1]);
        struct ggml_tensor * v_perm = ggml_permute(ctx, v_reshaped, 0, 2, 1, 3);

        // 6. Flash Attention
        float scale_val = 1.0f / std::sqrt((float)head_dim);
        struct ggml_tensor * attn_out_perm = ggml_flash_attn_ext(ctx, q_perm, k_perm, v_perm, mask, scale_val, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out_perm, GGML_PREC_F32);

        // 7. Flatten back to [n_embd, q_len]
        struct ggml_tensor * attn_out = ggml_reshape_2d(ctx, attn_out_perm, config.n_embd, q->ne[1]);

        // 8. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 9. Residual connection
        cur = ggml_add(ctx, cur, attn_proj);

        // 10. FFN RMSNorm
        h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.ffn_norm);

        // 11. FFN SwiGLU
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, h);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, h);

        struct ggml_tensor * gate_silu = ggml_silu(ctx, gate);
        struct ggml_tensor * ffn_out   = ggml_mul(ctx, gate_silu, up);
        struct ggml_tensor * ffn_proj  = ggml_mul_mat(ctx, layer.ffn_down, ffn_out);

        // 12. Residual connection
        cur = ggml_add(ctx, cur, ffn_proj);
    }

    // 13. Final RMSNorm
    struct ggml_tensor * final_node = cur;
    if (need_logits) {
        struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());

        // Extract ONLY the last token from final_norm: shape [n_embd, 1]
        int chunk_len = input_tokens->ne[0];
        struct ggml_tensor * last_token_norm = ggml_view_1d(ctx, final_norm, config.n_embd, (chunk_len - 1) * config.n_embd * sizeof(float));

        // Multiply last_token_norm [n_embd, 1] by model.get_output() [n_embd, n_vocab] -> logits [n_vocab, 1]
        struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), last_token_norm);
        *out_logits = logits;
        final_node = logits;
    } else {
        if (out_logits) *out_logits = nullptr;
    }

    ggml_build_forward_expand(gf, final_node);

    // Add views of k_layers and v_layers to prevent graph allocator reuse
    if (k_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*k_layers)[l]) {
                struct ggml_tensor * dummy_k = ggml_view_1d(ctx, (*k_layers)[l], (*k_layers)[l]->ne[0] * (*k_layers)[l]->ne[1], 0);
                ggml_build_forward_expand(gf, dummy_k);
            }
        }
    }
    if (v_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*v_layers)[l]) {
                struct ggml_tensor * dummy_v = ggml_view_1d(ctx, (*v_layers)[l], (*v_layers)[l]->ne[0] * (*v_layers)[l]->ne[1], 0);
                ggml_build_forward_expand(gf, dummy_v);
            }
        }
    }

    return gf;
}

// Helper to build the Qwen 2.5 sparse decode forward pass graph with SRL routing and custom Metal attention
struct ggml_cgraph * build_decode_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_token,
    struct ggml_tensor * position,
    struct ggml_tensor * W_proj,
    struct ggml_tensor * desc_matrix, // Layer 0 desc_matrix for SRL routing
    struct ggml_tensor * anchors_K,   // Layer 0 anchors_K for SRL routing
    struct ggml_tensor * slots_mask,  // slots_mask to ignore unoccupied slots
    struct ggml_tensor * host_slots,  // Host-computed candidate slots
    int srl_k_semantic,
    int srl_k_keep,
    CustomAttnUserData * userdata,     // Array of size config.n_layer!
    struct ggml_tensor ** out_logits,
    struct ggml_tensor ** out_selected_slots,
    std::vector<struct ggml_tensor *>* k_layers = nullptr,
    std::vector<struct ggml_tensor *>* v_layers = nullptr
) {
    const auto & config = model.get_config();
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // 1. Embedding lookup: shape [n_embd, 1]
    struct ggml_tensor * cur = ggml_get_rows(ctx, model.get_token_embd(), input_token);

    struct ggml_tensor * selected_slots = nullptr;

    for (int l = 0; l < config.n_layer; ++l) {
        const auto & layer = model.get_layers()[l];

        // 2. Attention RMSNorm
        struct ggml_tensor * h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.attn_norm);

        // 3. QKV Projections
        struct ggml_tensor * q = ggml_mul_mat(ctx, layer.wq, h);
        if (layer.bq) q = ggml_add(ctx, q, layer.bq);
        struct ggml_tensor * k = ggml_mul_mat(ctx, layer.wk, h);
        if (layer.bk) k = ggml_add(ctx, k, layer.bk);
        struct ggml_tensor * v = ggml_mul_mat(ctx, layer.wv, h);
        if (layer.bv) v = ggml_add(ctx, v, layer.bv);

        // Export raw key/value tensors of each layer (before RoPE)
        if (k_layers) (*k_layers)[l] = k;
        if (v_layers) (*v_layers)[l] = v;

        // ── SRL Routing Pipeline at Layer 0 ──
        if (l == 0) {
            int head_dim = config.n_embd / config.n_head;
            // Reshape Q: [896, 1] -> [head_dim, n_head] = [64, 14]
            struct ggml_tensor * Q = ggml_reshape_2d(ctx, q, head_dim, config.n_head);

            // Compute query descriptor: [desc_dim, 1]
            struct ggml_tensor * q_desc = compute_query_desc(ctx, Q, W_proj);

            // Semantic search topk: [srl_k_semantic, 1]
            struct ggml_tensor * sem_slots = semantic_search_topk(ctx, q_desc, desc_matrix, slots_mask, srl_k_semantic);
            struct ggml_tensor * sem_slots_1d = ggml_reshape_1d(ctx, sem_slots, srl_k_semantic);

            // Concatenate semantic and host slots
            struct ggml_tensor * candidate_slots = ggml_concat(ctx, sem_slots_1d, host_slots, 0);

            // Anchor screening: [srl_k_keep]
            float scale = 1.0f / std::sqrt((float)head_dim);
            selected_slots = ggml_cont(ctx, anchor_screen(ctx, Q, anchors_K, candidate_slots, scale, srl_k_keep));

            // Save selected slots to out parameter
            if (out_selected_slots) {
                *out_selected_slots = selected_slots;
            }

            // Make sure the selected slots are computed in the graph
            ggml_build_forward_expand(gf, selected_slots);
        }

        // Apply RoPE to Q for the custom Metal kernel!
        int head_dim = config.n_embd / config.n_head;
        struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim, config.n_head, 1);
        struct ggml_tensor * q_rope = ggml_rope_ext(ctx, q_reshaped, position, nullptr, config.n_rot, GGML_ROPE_TYPE_NEOX, config.n_ctx, config.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        struct ggml_tensor * q_rope_flat = ggml_reshape_1d(ctx, q_rope, config.n_embd);

        // Debug graph build prints removed (moved to stderr, only if needed)

        struct ggml_tensor * attn_out = nullptr;
        if (userdata && selected_slots) {
            struct ggml_tensor * kv_concat = ggml_concat(ctx, k, v, 0);
            // Reconstruct attention output using the custom Metal kernel!
            struct ggml_tensor * custom_attn = ggml_map_custom3(
                ctx, q_rope_flat, selected_slots, kv_concat,
                custom_attention_op_callback, 1, &userdata[l]
            );
            attn_out = custom_attn;
        } else {
            // Fallback placeholder attention
            struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, config.n_embd / config.n_head, 1, config.n_head_kv);
            int group_size = config.n_head / config.n_head_kv;
            struct ggml_tensor * target_repeat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.n_embd / config.n_head, group_size, config.n_head_kv);
            struct ggml_tensor * v_repeated = ggml_repeat(ctx, v_reshaped, target_repeat);
            attn_out = ggml_reshape_1d(ctx, v_repeated, config.n_embd);
        }

        // 5. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 6. Residual connection
        cur = ggml_add(ctx, cur, attn_proj);

        // 7. FFN RMSNorm
        h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.ffn_norm);

        // 8. FFN SwiGLU: (SiLU(gate) * up) * down
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, h);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, h);

        struct ggml_tensor * gate_silu = ggml_silu(ctx, gate);
        struct ggml_tensor * ffn_out   = ggml_mul(ctx, gate_silu, up);
        struct ggml_tensor * ffn_proj  = ggml_mul_mat(ctx, layer.ffn_down, ffn_out);

        // 9. Residual connection
        cur = ggml_add(ctx, cur, ffn_proj);
    }

    // 10. Final RMSNorm
    struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
    final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());

    // 11. LM Head Output
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), final_norm);

    *out_logits = logits;
    ggml_build_forward_expand(gf, logits);

    if (out_selected_slots && *out_selected_slots) {
        struct ggml_tensor * dummy_slots = ggml_view_1d(ctx, *out_selected_slots, (*out_selected_slots)->ne[0], 0);
        ggml_build_forward_expand(gf, dummy_slots);
    }

    if (k_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*k_layers)[l]) {
                struct ggml_tensor * dummy_k = ggml_view_1d(ctx, (*k_layers)[l], (*k_layers)[l]->ne[0], 0);
                ggml_build_forward_expand(gf, dummy_k);
            }
        }
    }
    if (v_layers) {
        for (int l = 0; l < config.n_layer; ++l) {
            if ((*v_layers)[l]) {
                struct ggml_tensor * dummy_v = ggml_view_1d(ctx, (*v_layers)[l], (*v_layers)[l]->ne[0], 0);
                ggml_build_forward_expand(gf, dummy_v);
            }
        }
    }

    return gf;
}

bool verify_attention_cpu(
    const float* q_data,              // [n_q_heads * D]
    const int32_t* slots,             // [K]
    const float* metal_output,        // [n_q_heads * D]
    NativeBlockPool* kv_engine,
    int n_q_heads, int n_kv_heads, int rank, int S_max, int K, int D, float scale
) {
    // Read pool tensors to host for reference calculation
    std::vector<int8_t> U(ggml_nelements(kv_engine->get_U()));
    ggml_backend_tensor_get(kv_engine->get_U(), U.data(), 0, U.size() * sizeof(int8_t));

    std::vector<ggml_fp16_t> U_scale(ggml_nelements(kv_engine->get_U_scale()));
    ggml_backend_tensor_get(kv_engine->get_U_scale(), U_scale.data(), 0, U_scale.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> VK(ggml_nelements(kv_engine->get_VK()));
    ggml_backend_tensor_get(kv_engine->get_VK(), VK.data(), 0, VK.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> VV(ggml_nelements(kv_engine->get_VV()));
    ggml_backend_tensor_get(kv_engine->get_VV(), VV.data(), 0, VV.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> anchors_K(ggml_nelements(kv_engine->get_anchors_K()));
    ggml_backend_tensor_get(kv_engine->get_anchors_K(), anchors_K.data(), 0, anchors_K.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> anchors_V(ggml_nelements(kv_engine->get_anchors_V()));
    ggml_backend_tensor_get(kv_engine->get_anchors_V(), anchors_V.data(), 0, anchors_V.size() * sizeof(ggml_fp16_t));

    std::vector<int32_t> seq_lens(ggml_nelements(kv_engine->get_seq_lens()));
    ggml_backend_tensor_get(kv_engine->get_seq_lens(), seq_lens.data(), 0, seq_lens.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> scales(ggml_nelements(kv_engine->get_scales()));
    ggml_backend_tensor_get(kv_engine->get_scales(), scales.data(), 0, scales.size() * sizeof(ggml_fp16_t));

    std::vector<float> cpu_output(n_q_heads * D, 0.0f);
    const int g = n_q_heads / n_kv_heads;

    for (int h = 0; h < n_q_heads; ++h) {
        int kv_head = h / g;

        float max_score = -1e30f;
        
        struct SlotScoreInfo {
            float anchor_score;
            std::vector<float> token_scores;
            std::vector<float> q_proj;
        };
        std::vector<SlotScoreInfo> slot_infos(K);

        for (int k = 0; k < K; ++k) {
            int slot_id = slots[k];
            int slen = seq_lens[slot_id];
            float scale_u = ggml_fp16_to_fp32(U_scale[slot_id]);
            float block_scale = ggml_fp16_to_fp32(scales[slot_id]);

            // 1. Anchor score
            float score_anc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float q_val = q_data[h * D + d];
                float ak_val = ggml_fp16_to_fp32(anchors_K[slot_id * n_kv_heads * D + kv_head * D + d]);
                score_anc += q_val * ak_val;
            }
            slot_infos[k].anchor_score = score_anc;

            float s_anc_scaled = score_anc * scale;
            if (s_anc_scaled > max_score) max_score = s_anc_scaled;

            // 2. Query projection
            std::vector<float> q_proj(rank, 0.0f);
            for (int r = 0; r < rank; ++r) {
                float proj = 0.0f;
                int base_vk_offset = slot_id * rank * n_kv_heads * D + r * n_kv_heads * D + kv_head * D;
                for (int d = 0; d < D; ++d) {
                    float q_val = q_data[h * D + d];
                    float vk_val = ggml_fp16_to_fp32(VK[base_vk_offset + d]);
                    proj += q_val * vk_val;
                }
                q_proj[r] = proj;
            }
            slot_infos[k].q_proj = q_proj;

            // 3. Token scores
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

        // Compute softmax denominator
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k) {
            sum_exp += std::exp(slot_infos[k].anchor_score * scale - max_score);
            for (float s : slot_infos[k].token_scores) {
                sum_exp += std::exp(s - max_score);
            }
        }

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

    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output[i] - metal_output[i]);
        if (diff > max_diff) max_diff = diff;
        sum_sq_diff += diff * diff;
    }
    float rmse = std::sqrt(sum_sq_diff / cpu_output.size());
    std::printf("[Verification] CPU vs Metal Max Diff: %e, RMSE: %e\n", max_diff, rmse);
    
    if (max_diff < 1e-4f) {
        std::printf("[Verification] SUCCESS: Metal attention matches CPU reference!\n");
        return true;
    } else {
        std::printf("[Verification] FAILURE: Metal attention does not match CPU reference!\n");
        return false;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path> [prompt]" << std::endl;
        return 1;
    }

    // ── Initialize Best Backend (GPU / CPU) ──────────────────────────────────
    ggml_backend_owner backend_owner;
    if (!backend_owner.sched) {
        std::cerr << "Failed to initialize backend scheduler!" << std::endl;
        return 1;
    }
    std::cerr << "[DiffKV Native] Initialized scheduler with backends: GPU=" 
              << (backend_owner.gpu_backend ? ggml_backend_name(backend_owner.gpu_backend) : "none") 
              << ", CPU=" << (backend_owner.cpu_backend ? ggml_backend_name(backend_owner.cpu_backend) : "none") << std::endl;

    ggml_backend_t backend = backend_owner.gpu_backend;
    ggml_backend_sched_t sched = backend_owner.sched;

    std::string model_path = argv[1];
    diffkv::DiffKVModel model;

    if (!model.load_from_file(model_path, backend)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    model.print_info();
    
    // ── SRL configuration (with defaults) ───────────────────────────────────
    int srl_k_semantic = 32;
    int srl_k_lexical = 8;
    int srl_k_graph = 8;
    int srl_k_recency = 8;
    int srl_k_keep = 16;

    // Load from env vars
    if (const char* env = std::getenv("DIFFKV_SRL_K_SEM")) srl_k_semantic = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_LEX")) srl_k_lexical = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_GRAPH")) srl_k_graph = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_RECENCY")) srl_k_recency = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_KEEP")) srl_k_keep = std::stoi(env);

    // Also parse from argv
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--srl-k-semantic" && i + 1 < argc) {
            srl_k_semantic = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-lexical" && i + 1 < argc) {
            srl_k_lexical = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-graph" && i + 1 < argc) {
            srl_k_graph = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-recency" && i + 1 < argc) {
            srl_k_recency = std::stoi(argv[++i]);
        } else if (arg == "--srl-k-keep" && i + 1 < argc) {
            srl_k_keep = std::stoi(argv[++i]);
        }
    }

    int srl_k_host = 1 + srl_k_recency + srl_k_lexical + 2 * srl_k_lexical + srl_k_graph;

    std::unordered_set<int32_t> stop_token_ids;
    {
        std::vector<std::string> stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "need",
            "to", "of", "in", "on", "at", "by", "for", "with", "as",
            "and", "or", "but", "if", "then", "that", "this", "it",
            "he", "she", "they", "we", "you", "i", "not", "no",
            ",", ".", ":", ";", "?", "!", "(", ")", "'", "\"", "-", "\n",
            "system", "user", "assistant", "im_start", "im_end"
        };
        for (const auto & word : stop_words) {
            auto t = model.tokenize(word, false);
            for (int32_t tok : t) {
                stop_token_ids.insert(tok);
            }
        }
        for (int i = 0; i < 200; ++i) {
            stop_token_ids.insert(i);
        }
    }

    diffkv::InvertedIndex inverted_index;


    // Backend is already initialized at the start of main
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Initialize NativeBlockPool block pool for all layers
    int n_slots = model.get_config().n_ctx / 64;
    if (const char* env_slots = std::getenv("DIFFKV_MAX_CONTEXT_SLOTS")) {
        n_slots = std::stoi(env_slots);
        std::cerr << "[DiffKV Native] Overriding n_slots from DIFFKV_MAX_CONTEXT_SLOTS: " << n_slots << std::endl;
    }
    srl_k_semantic = std::min(srl_k_semantic, n_slots);
    srl_k_keep = std::min(srl_k_keep, n_slots);
    int rank = 32;
    int head_dim = model.get_config().n_embd / model.get_config().n_head;
    int kv_heads = model.get_config().n_head_kv;
    int desc_dim = 64;
    int n_vocab = model.get_config().n_vocab;
    int n_layers = model.get_config().n_layer;

    int micro_block_size = 16;
    if (const char* env_mbs = std::getenv("DIFFKV_MICRO_BLOCK_SIZE")) {
        micro_block_size = std::stoi(env_mbs);
    }
    float gpu_budget_gb = 2.0f;
    if (const char* env_budget = std::getenv("DIFFKV_GPU_BUDGET_GB")) {
        gpu_budget_gb = std::stof(env_budget);
    }
    size_t gpu_budget_bytes = static_cast<size_t>(gpu_budget_gb * 1024.0f * 1024.0f * 1024.0f);

    diffkv::KVRuntimeManager runtime_manager(rank, micro_block_size, gpu_budget_bytes);
    if (!runtime_manager.initialize(n_slots, head_dim, kv_heads, desc_dim, n_layers, &model, buft)) {
        std::cerr << "Failed to initialize KVRuntimeManager!" << std::endl;
        return 1;
    }
    auto & kv_engines = runtime_manager.get_engines();

    // Initialize W_proj on host
    srand(42);
    std::vector<float> W_proj_host(head_dim * desc_dim);
    for (int r = 0; r < desc_dim; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < head_dim; ++c) {
            float val = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            W_proj_host[r * head_dim + c] = val;
            sum_sq += val * val;
        }
        float norm = std::sqrt(sum_sq) + 1e-8f;
        for (int c = 0; c < head_dim; ++c) {
            W_proj_host[r * head_dim + c] /= norm;
        }
    }

    // Clear pool tensors for all layers
    std::vector<float> desc_matrix_host(desc_dim * n_slots, 0.0f);
    std::vector<ggml_fp16_t> anchors_K_host(head_dim * kv_heads * n_slots, ggml_fp32_to_fp16(0.0f));
    std::vector<int8_t> U_host(rank * 64 * n_slots, 0);
    std::vector<ggml_fp16_t> U_scale_host(n_slots, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> VK_host(head_dim * kv_heads * rank * n_slots, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> VV_host(head_dim * kv_heads * rank * n_slots, ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> anchors_V_host(head_dim * kv_heads * n_slots, ggml_fp32_to_fp16(0.0f));
    std::vector<int32_t> seq_lens_host(n_slots, 0);
    std::vector<ggml_fp16_t> scales_host(n_slots, ggml_fp32_to_fp16(0.0f));
    std::vector<int32_t> anchor_positions_host(n_slots, 0);  // Actual sequence position of each block's anchor

    for (int l = 0; l < n_layers; ++l) {
        auto & engine = *kv_engines[l];
        ggml_backend_tensor_set(engine.get_desc_matrix(), desc_matrix_host.data(), 0, desc_matrix_host.size() * sizeof(float));
        ggml_backend_tensor_set(engine.get_anchors_K(), anchors_K_host.data(), 0, anchors_K_host.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine.get_U(), U_host.data(), 0, U_host.size() * sizeof(int8_t));
        ggml_backend_tensor_set(engine.get_U_scale(), U_scale_host.data(), 0, U_scale_host.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine.get_VK(), VK_host.data(), 0, VK_host.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine.get_VV(), VV_host.data(), 0, VV_host.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine.get_anchors_V(), anchors_V_host.data(), 0, anchors_V_host.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine.get_seq_lens(), seq_lens_host.data(), 0, seq_lens_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(engine.get_scales(), scales_host.data(), 0, scales_host.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(engine.get_anchor_positions(), anchor_positions_host.data(), 0, anchor_positions_host.size() * sizeof(int32_t));
    }

    // Reference SVD compressor from runtime_manager
    auto & compressor = runtime_manager.get_compressor();

    // Allocate persistent dense vectors and tables
    int F_test = kv_heads * head_dim;
    std::vector<std::vector<float>> active_k_dense(n_layers, std::vector<float>(64 * F_test, 0.0f));
    std::vector<std::vector<float>> active_v_dense(n_layers, std::vector<float>(64 * F_test, 0.0f));
    std::map<int, std::vector<float>> persistent_k_dense;
    std::map<int, std::vector<float>> persistent_v_dense;
    std::vector<std::vector<int32_t>> seq_lens_by_layer(n_layers, std::vector<int32_t>(n_slots, 0));



    bool interactive = (argc < 3 || std::string(argv[2]) == "-");

    while (true) {
        std::string prompt;
        if (interactive) {
            std::cout << "__READY__" << std::endl;
            if (!std::getline(std::cin, prompt)) {
                break;
            }
            if (prompt.empty() || prompt == "exit" || prompt == "quit") {
                break;
            }
            // Unescape \\n sequences back to real newlines (encoded by gateway for single-line stdin)
            {
                std::string unescaped;
                unescaped.reserve(prompt.size());
                for (size_t ui = 0; ui < prompt.size(); ++ui) {
                    if (ui + 1 < prompt.size() && prompt[ui] == '\\' && prompt[ui+1] == 'n') {
                        unescaped += '\n';
                        ++ui;
                    } else {
                        unescaped += prompt[ui];
                    }
                }
                prompt = std::move(unescaped);
            }
        } else {
            prompt = argv[2];
        }

        // Reset runtime manager and helper structures
        runtime_manager.reset();

        for (int l = 0; l < n_layers; ++l) {
            std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
            std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
        }
        persistent_k_dense.clear();
        persistent_v_dense.clear();
        inverted_index.clear();
        for (int l = 0; l < n_layers; ++l) {
            std::fill(seq_lens_by_layer[l].begin(), seq_lens_by_layer[l].end(), 0);
        }

        // Wrap prompt in Qwen2.5 instruction chat template (skip if gateway already formatted it)
        std::string chat_prompt;
        if (prompt.rfind("<|im_start|", 0) == 0) {
            // Gateway sent a pre-formatted multi-turn conversation — use as-is
            chat_prompt = prompt;
            std::cerr << "[DiffKV Native] Using pre-formatted chat prompt (" << prompt.size() << " bytes)" << std::endl;
        } else {
            chat_prompt =
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                "<|im_start|>assistant\n";
        }
        
        if (!interactive) {
            std::cerr << "[DiffKV Native] Prompt: \"" << prompt << "\"" << std::endl;
        }

        std::vector<int32_t> prompt_tokens = model.tokenize(chat_prompt, true);
        if (prompt_tokens.empty()) {
            std::cerr << "Error: Tokenization returned empty list!" << std::endl;
            if (!interactive) break;
            continue;
        }

        if (!interactive) {
            std::cerr << "[DiffKV Native] Prompt tokens (" << prompt_tokens.size() << "): ";
            for (int32_t t : prompt_tokens) std::cerr << t << " ";
            std::cerr << "\n";
        }

        int L = prompt_tokens.size();
        if (L > n_slots * 64) {
            std::cerr << "[DiffKV Native] Warning: Prompt tokens length " << L 
                      << " exceeds maximum capacity " << n_slots * 64 
                      << ". Truncating prompt." << std::endl;
            L = n_slots * 64;
            prompt_tokens.resize(L);
        }

        // ── 1. PREFILL PHASE ──
        if (!interactive) {
            std::cerr << "[DiffKV Native] Running Prefill phase in chunks..." << std::endl;
        }

        std::vector<std::vector<float>> k_activations(n_layers, std::vector<float>(L * F_test));
        std::vector<std::vector<float>> v_activations(n_layers, std::vector<float>(L * F_test));
        std::vector<float> prefill_output_logits(n_vocab);

        int chunk_size = 2048;  // Larger chunks = fewer graph alloc/compute cycles = faster prefill
        int pos_start = 0;

        while (pos_start < L) {
            int chunk_len = std::min(chunk_size, L - pos_start);

            struct ggml_init_params prefill_params = {
                /*.mem_size   =*/ 4 * 1024 * 1024,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * prefill_ctx = ggml_init(prefill_params);
            if (!prefill_ctx) {
                std::cerr << "Failed to initialize prefill context!" << std::endl;
                break;
            }

            struct ggml_tensor * input_tokens_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
            ggml_set_input(input_tokens_prefill);
            struct ggml_tensor * positions_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
            ggml_set_input(positions_prefill);
            struct ggml_tensor * mask_prefill = ggml_new_tensor_2d(prefill_ctx, GGML_TYPE_F16, chunk_len, chunk_len);
            ggml_set_input(mask_prefill);

            struct ggml_tensor * prefill_logits = nullptr;
            std::vector<struct ggml_tensor *> prefill_k_layers(n_layers, nullptr);
            std::vector<struct ggml_tensor *> prefill_v_layers(n_layers, nullptr);
            bool is_last_chunk = (pos_start + chunk_len >= L);
            struct ggml_cgraph * prefill_graph = build_prefill_graph(
                prefill_ctx, model, input_tokens_prefill, positions_prefill, mask_prefill,
                &prefill_logits, &prefill_k_layers, &prefill_v_layers, is_last_chunk
            );
            if (is_last_chunk && prefill_logits) {
                ggml_set_output(prefill_logits);
            }

            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, prefill_graph)) {
                std::cerr << "Failed to allocate memory for prefill graph!" << std::endl;
                ggml_free(prefill_ctx);
                break;
            }

            ggml_backend_tensor_set(input_tokens_prefill, prompt_tokens.data() + pos_start, 0, chunk_len * sizeof(int32_t));

            std::vector<int32_t> pos_host(chunk_len);
            for (int i = 0; i < chunk_len; ++i) pos_host[i] = pos_start + i;
            ggml_backend_tensor_set(positions_prefill, pos_host.data(), 0, chunk_len * sizeof(int32_t));

            std::vector<ggml_fp16_t> mask_host(chunk_len * chunk_len, ggml_fp32_to_fp16(0.0f));
            for (int i = 0; i < chunk_len; ++i) {
                for (int j = i + 1; j < chunk_len; ++j) {
                    mask_host[i * chunk_len + j] = ggml_fp32_to_fp16(-INFINITY);
                }
            }
            ggml_backend_tensor_set(mask_prefill, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));

            if (ggml_backend_sched_graph_compute(sched, prefill_graph) != GGML_STATUS_SUCCESS) {
                std::cerr << "Error: Prefill graph compute failed!" << std::endl;
                ggml_free(prefill_ctx);
                break;
            }

            std::vector<std::vector<float>> chunk_k(n_layers, std::vector<float>(chunk_len * F_test));
            std::vector<std::vector<float>> chunk_v(n_layers, std::vector<float>(chunk_len * F_test));
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_get(prefill_k_layers[l], chunk_k[l].data(), 0, chunk_len * F_test * sizeof(float));
                ggml_backend_tensor_get(prefill_v_layers[l], chunk_v[l].data(), 0, chunk_len * F_test * sizeof(float));
                std::memcpy(k_activations[l].data() + pos_start * F_test, chunk_k[l].data(), chunk_len * F_test * sizeof(float));
                std::memcpy(v_activations[l].data() + pos_start * F_test, chunk_v[l].data(), chunk_len * F_test * sizeof(float));
            }
            runtime_manager.ingest_prefill(chunk_k, chunk_v, chunk_len, pos_start, prompt_tokens);

            if (pos_start + chunk_len >= L && prefill_logits) {
                ggml_backend_tensor_get(prefill_logits, prefill_output_logits.data(), 0, n_vocab * sizeof(float));
            }

            ggml_free(prefill_ctx);

            pos_start += chunk_len;
        }

        if (!interactive) {
            std::cerr << "[DiffKV Native] Waiting for background SVD compressor to catch up..." << std::endl;
        }
        runtime_manager.wait_for_compressor();
        runtime_manager.update_descriptors(W_proj_host, desc_dim, head_dim);

        // Logits already retrieved inside the chunk loop

        std::vector<std::pair<float, int>> prefill_top_k;
        for (int i = 0; i < n_vocab; ++i) {
            prefill_top_k.push_back({prefill_output_logits[i], i});
        }
        std::sort(prefill_top_k.begin(), prefill_top_k.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

        if (!interactive) {
            std::cerr << "\n[Prefill Phase Top predictions]:\n";
            for (int k = 0; k < 5; ++k) {
                std::cerr << "  " << k << ": \"" << model.token_to_piece(prefill_top_k[k].second) << "\" (id: " << prefill_top_k[k].second << ", logit: " << prefill_top_k[k].first << ")\n";
            }
        }

        int32_t first_decode_token = prefill_top_k[0].second;

        // Resources already freed inside the chunk loop

        // ── 2. DECODE PHASE — rebuild decode graph fresh (avoids sched-ctx pointer corruption) ──
        std::cerr << "[DiffKV Native] Building fresh decode graph..." << std::flush;
        struct ggml_init_params decode_params = {
            /*.mem_size   =*/ 4 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * decode_ctx = ggml_init(decode_params);
        if (!decode_ctx) {
            std::cerr << "\n[ERROR] Failed to initialize decode context!" << std::endl;
            if (!interactive) break; else continue;
        }

        struct ggml_tensor * input_token_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(input_token_decode);
        struct ggml_tensor * position_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(position_decode);
        struct ggml_tensor * W_proj_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F32, head_dim, desc_dim);
        ggml_set_input(W_proj_decode);
        struct ggml_tensor * slots_mask_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_F32, n_slots);
        ggml_set_input(slots_mask_decode);
        struct ggml_tensor * host_slots_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, srl_k_host);
        ggml_set_input(host_slots_decode);

        std::vector<diffkv::CustomAttnUserData> userdata(n_layers);
        for (int l = 0; l < n_layers; ++l) {
            userdata[l].kv_engine = kv_engines[l].get();
            userdata[l].slot_indices = nullptr;
            userdata[l].n_q_heads = model.get_config().n_head;
            userdata[l].n_kv_heads = model.get_config().n_head_kv;
            userdata[l].rank = rank;
            userdata[l].S_max = 64;
            userdata[l].K = 0;
            userdata[l].D = head_dim;
            userdata[l].scale = 1.0f / std::sqrt((float)head_dim);
            userdata[l].has_rope = true;
            userdata[l].rope_freq_base = model.get_config().rope_freq_base;
        }

        struct ggml_tensor * decode_logits = nullptr;
        struct ggml_tensor * decode_selected_slots = nullptr;
        std::vector<struct ggml_tensor *> decode_k_layers(n_layers, nullptr);
        std::vector<struct ggml_tensor *> decode_v_layers(n_layers, nullptr);

        struct ggml_cgraph * decode_graph = build_decode_graph(
            decode_ctx, model, input_token_decode, position_decode, W_proj_decode,
            kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
            slots_mask_decode, host_slots_decode,
            srl_k_semantic, srl_k_keep,
            userdata.data(), &decode_logits, &decode_selected_slots,
            &decode_k_layers, &decode_v_layers
        );
        ggml_set_output(decode_logits);
        if (decode_selected_slots) ggml_set_output(decode_selected_slots);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, decode_graph)) {
            std::cerr << "\n[ERROR] Failed to allocate decode graph — skipping prompt." << std::endl;
            ggml_free(decode_ctx);
            if (!interactive) break; else continue;
        }
        ggml_backend_tensor_set(W_proj_decode, W_proj_host.data(), 0, W_proj_host.size() * sizeof(float));
        std::cerr << " OK" << std::endl;

        if (interactive) {
            std::cout << "__RESPONSE__" << std::endl;
        } else {
            std::cout << "[Response] " << std::flush;
        }

        int32_t last_token = first_decode_token;
        std::string first_piece = model.token_to_piece(last_token);
        std::cout << first_piece << std::flush;

        std::vector<int32_t> generated_tokens;
        generated_tokens.push_back(last_token);
        // max_generate: default 2048, overridable via DIFFKV_MAX_TOKENS env var or request
        int max_generate = 2048;
        if (const char* env_mt = std::getenv("DIFFKV_MAX_TOKENS")) {
            max_generate = std::max(1, std::stoi(env_mt));
        }

        std::vector<int32_t> all_tokens = prompt_tokens;
        all_tokens.push_back(last_token);

        // Populate initial inverted index for all completed blocks in prefill using runtime_manager
        inverted_index.clear();
        auto & blocks_layer0 = runtime_manager.get_ingest_manager().get_blocks(0);
        int completed_blocks = blocks_layer0.size();
        for (int i = 0; i < completed_blocks; ++i) {
            if (i < completed_blocks - 1 || blocks_layer0[i]->token_count() == 1 + micro_block_size) {
                inverted_index.add_block_tokens(i, blocks_layer0[i]->token_indices, blocks_layer0[i]->anchor_idx, stop_token_ids);
            }
        }
        inverted_index.recompute_idf(completed_blocks);

        int active_slot = runtime_manager.get_ingest_manager().get_blocks(0).size() - 1;
        if (active_slot < 0) {
            active_slot = 0;
        }
        int active_block_tokens = 0;
        if (!runtime_manager.get_ingest_manager().get_blocks(0).empty()) {
            active_block_tokens = runtime_manager.get_ingest_manager().get_blocks(0).back()->token_count();
            if (active_block_tokens > 0) {
                active_block_tokens--; // exclude anchor
            }
        }

        for (int l = 0; l < n_layers; ++l) {
            auto & b_list = runtime_manager.get_ingest_manager().get_blocks(l);
            if (active_block_tokens > 0 && !b_list.empty()) {
                auto & last_block = b_list.back();
                std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
                std::memcpy(active_k_dense[l].data(), last_block->active_k.data(), last_block->active_k.size() * sizeof(float));
                std::memcpy(active_v_dense[l].data(), last_block->active_v.data(), last_block->active_v.size() * sizeof(float));
            } else {
                std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
            }
        }

        for (int step = 0; step < max_generate; ++step) {
            if (active_slot >= n_slots) {
                std::cerr << "\n[DiffKV Native] Warning: Context slot capacity reached during decode. Stopping generation." << std::endl;
                break;
            }
            int current_pos = L + step;

            ggml_backend_tensor_set(input_token_decode, &last_token, 0, sizeof(last_token));
            ggml_backend_tensor_set(position_decode, &current_pos, 0, sizeof(current_pos));

            // ── Host Candidate Selection (SRL) ──────────────────────────────────────
            std::vector<int32_t> routed_blocks = runtime_manager.route_decode_slots(
                current_pos,
                all_tokens,
                inverted_index,
                stop_token_ids,
                srl_k_recency,
                srl_k_lexical,
                srl_k_graph,
                srl_k_host,
                active_slot
            );
            
            // Touch blocks to ensure CPU Resident blocks are reloaded to GPU
            runtime_manager.touch_active_slots(routed_blocks);
            
            // Translate block indices to physical pool slot indices
            std::vector<int32_t> physical_candidates;
            for (int32_t block_idx : routed_blocks) {
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                if (block_idx >= 0 && block_idx < (int)blocks.size()) {
                    physical_candidates.push_back(blocks[block_idx]->pool_idx);
                } else {
                    physical_candidates.push_back(0);
                }
            }
            
            ggml_backend_tensor_set(host_slots_decode, physical_candidates.data(), 0, srl_k_host * sizeof(int32_t));

            std::vector<float> slots_mask_host(n_slots, -1e10f);
            int occupied_up_to = active_slot - 1;
            for (int i = 0; i <= occupied_up_to; ++i) {
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                if (i >= 0 && i < (int)blocks.size()) {
                    int slot_id = blocks[i]->pool_idx;
                    if (slot_id >= 0 && slot_id < n_slots) {
                        slots_mask_host[slot_id] = 0.0f;
                    }
                }
            }
            ggml_backend_tensor_set(slots_mask_decode, slots_mask_host.data(), 0, n_slots * sizeof(float));

            int current_k = std::max(0, std::min(srl_k_keep, active_slot));
            
            // Get physical active slot index
            int physical_active_slot = 0;
            auto & b_list = runtime_manager.get_ingest_manager().get_blocks(0);
            if (active_slot >= 0 && active_slot < (int)b_list.size()) {
                physical_active_slot = b_list[active_slot]->pool_idx;
            }

            for (int l = 0; l < n_layers; ++l) {
                userdata[l].K = current_k;
                userdata[l].active_k_dense = active_k_dense[l].data();
                userdata[l].active_v_dense = active_v_dense[l].data();
                userdata[l].active_block_tokens = active_block_tokens;
                userdata[l].active_slot = physical_active_slot;
            }


            if (ggml_backend_sched_graph_compute(sched, decode_graph) != GGML_STATUS_SUCCESS) {
                std::cerr << "Error: Decode graph compute failed at step " << step << std::endl;
                break;
            }

            std::vector<float> output_logits(n_vocab);
            ggml_backend_tensor_get(decode_logits, output_logits.data(), 0, n_vocab * sizeof(float));

            constexpr float REP_PENALTY = 1.15f;
            for (int32_t tok : generated_tokens) {
                if (tok >= 0 && tok < n_vocab) {
                    float& l = output_logits[tok];
                    l = (l > 0.0f) ? l / REP_PENALTY : l * REP_PENALTY;
                }
            }
            if (last_token >= 0 && last_token < n_vocab) {
                float& l = output_logits[last_token];
                l = (l > 0.0f) ? l / REP_PENALTY : l * REP_PENALTY;
            }

            std::vector<std::pair<float, int>> logits_sorted;
            for (int i = 0; i < n_vocab; ++i) {
                logits_sorted.push_back({output_logits[i], i});
            }
            std::sort(logits_sorted.begin(), logits_sorted.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

            if (!interactive) {
                std::cerr << "\n[Step " << step << " Top predictions]:\n";
                for (int i = 0; i < std::min(5, n_vocab); ++i) {
                    std::cerr << "  " << i << ": \"" << model.token_to_piece(logits_sorted[i].second) << "\" (id: " << logits_sorted[i].second << ", logit: " << logits_sorted[i].first << ")\n";
                }
            }

            int32_t next_token = logits_sorted[0].second;

            if (model.is_eog_token(next_token) || next_token == model.token_eos()) {
                if (!interactive) {
                    std::cerr << " [EOS]" << std::endl;
                }
                break;
            }

            std::vector<std::vector<float>> decode_k(n_layers, std::vector<float>(F_test));
            std::vector<std::vector<float>> decode_v(n_layers, std::vector<float>(F_test));
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_get(decode_k_layers[l], decode_k[l].data(), 0, F_test * sizeof(float));
                ggml_backend_tensor_get(decode_v_layers[l], decode_v[l].data(), 0, F_test * sizeof(float));
            }
            
            runtime_manager.ingest_decode(decode_k, decode_v, current_pos, all_tokens);

            int old_active_slot = active_slot;
            active_slot = runtime_manager.get_ingest_manager().get_blocks(0).size() - 1;
            if (active_slot < 0) {
                active_slot = 0;
            }
            active_block_tokens = 0;
            if (!runtime_manager.get_ingest_manager().get_blocks(0).empty()) {
                active_block_tokens = runtime_manager.get_ingest_manager().get_blocks(0).back()->token_count();
                if (active_block_tokens > 0) {
                    active_block_tokens--; // exclude anchor
                }
            }

            // Sync active dense buffers for custom attention
            for (int l = 0; l < n_layers; ++l) {
                auto & b_list = runtime_manager.get_ingest_manager().get_blocks(l);
                if (active_block_tokens > 0 && !b_list.empty()) {
                    auto & last_block = b_list.back();
                    std::memcpy(active_k_dense[l].data(), last_block->active_k.data(), last_block->active_k.size() * sizeof(float));
                    std::memcpy(active_v_dense[l].data(), last_block->active_v.data(), last_block->active_v.size() * sizeof(float));
                } else {
                    std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                    std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
                }
            }

            // Index completed block to inverted index
            if (active_block_tokens == 0 && active_slot > old_active_slot) {
                int prev_slot = old_active_slot;
                auto & blocks = runtime_manager.get_ingest_manager().get_blocks(0);
                if (prev_slot >= 0 && prev_slot < (int)blocks.size()) {
                    auto & block = blocks[prev_slot];
                    inverted_index.add_block_tokens(prev_slot, block->token_indices, block->anchor_idx, stop_token_ids);
                    inverted_index.recompute_idf(prev_slot + 1);
                }
                // Update semantic descriptors
                runtime_manager.update_descriptors(W_proj_host, desc_dim, head_dim);

                for (int l = 0; l < n_layers; ++l) {
                    std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                    std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
                }
            }
        }
        std::cout << std::endl;
        if (interactive) {
            std::cout << "__FINISH__" << std::endl;
        }

        // Free decode context — must happen before next iteration rebuilds it
        ggml_free(decode_ctx);

        if (!interactive) {
            break;
        }
    }

    // Stop compressor and cleanup
    compressor.stop();

    std::cerr << "[DiffKV Native] Text generation completed successfully!" << std::endl;
    return 0;
}
