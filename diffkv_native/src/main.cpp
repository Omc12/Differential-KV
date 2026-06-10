#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "diffkv_model.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "diffkv_kv_engine.hpp"
#include "diffkv_srl.hpp"
#include "diffkv_attention.hpp"
#include "diffkv_compressor.hpp"

using namespace diffkv;

// Helper to build the Qwen 2.5 dense prefill graph using causal flash attention
struct ggml_cgraph * build_prefill_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_tokens,
    struct ggml_tensor * positions,
    struct ggml_tensor * mask,
    struct ggml_tensor ** out_logits,
    std::vector<struct ggml_tensor *>* k_layers = nullptr,
    std::vector<struct ggml_tensor *>* v_layers = nullptr
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
    struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
    final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());

    // 14. LM Head Output
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), final_norm);

    *out_logits = logits;
    ggml_build_forward_expand(gf, logits);

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

            // Semantic search topk: [k_semantic, 1]
            int k_semantic = 16;
            struct ggml_tensor * candidate_slots = semantic_search_topk(ctx, q_desc, desc_matrix, slots_mask, k_semantic);

            // Anchor screening: [k_keep]
            float scale = 1.0f / std::sqrt((float)head_dim);
            int k_keep = 8;
            selected_slots = anchor_screen(ctx, Q, anchors_K, candidate_slots, scale, k_keep);

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

        struct ggml_tensor * attn_out = nullptr;
        if (userdata && selected_slots) {
            // Reconstruct attention output using the custom Metal kernel!
            struct ggml_tensor * custom_attn = ggml_map_custom2(
                ctx, q_rope_flat, selected_slots,
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
    DiffKVKVEngine* kv_engine,
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

    std::string model_path = argv[1];
    diffkv::DiffKVModel model;

    if (!model.load_from_file(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    model.print_info();

    // ── Set up CPU Backend ──────────────────────────────────────────────────
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::cerr << "Failed to initialize CPU backend!" << std::endl;
        return 1;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    // Initialize DiffKVKVEngine block pool for all layers
    int n_slots = 32;
    int rank = 32;
    int head_dim = model.get_config().n_embd / model.get_config().n_head;
    int kv_heads = model.get_config().n_head_kv;
    int desc_dim = 64;
    int n_vocab = model.get_config().n_vocab;
    int n_layers = model.get_config().n_layer;

    std::vector<std::unique_ptr<diffkv::DiffKVKVEngine>> kv_engines(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        kv_engines[l] = std::make_unique<diffkv::DiffKVKVEngine>();
        if (!kv_engines[l]->initialize(n_slots, rank, head_dim, kv_heads, desc_dim, buft)) {
            std::cerr << "Failed to initialize DiffKVKVEngine block pool for layer " << l << "!" << std::endl;
            ggml_backend_free(backend);
            return 1;
        }
    }

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

    // Initialize SVD compressor using Layer 0 state table
    diffkv::DiffKVCompressor compressor(kv_engines[0]->get_state_table());
    if (!compressor.start()) {
        std::cerr << "Error: Failed to start compressor thread!" << std::endl;
        ggml_backend_free(backend);
        return 1;
    }

    // Define prompt and tokenize
    std::string prompt = "Explain SVD.";
    if (argc >= 3) {
        prompt = argv[2];
    }
    // Wrap prompt in Qwen2.5 instruction chat template for proper instruction following
    std::string chat_prompt =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n";
    std::cout << "[DiffKV Native] Prompt: \"" << prompt << "\"" << std::endl;
    std::vector<int32_t> prompt_tokens = model.tokenize(chat_prompt, true);
    if (prompt_tokens.empty()) {
        std::cerr << "Error: Tokenization returned empty list!" << std::endl;
        compressor.stop();
        ggml_backend_free(backend);
        return 1;
    }

    std::cout << "[DiffKV Native] Prompt tokens (" << prompt_tokens.size() << "): ";
    for (int32_t t : prompt_tokens) std::cout << t << " ";
    std::cout << "\n";

    int L = prompt_tokens.size();
    int F_test = kv_heads * head_dim;

    // ── 1. PREFILL PHASE (Single pass causal attention) ──
    std::cout << "[DiffKV Native] Running Prefill phase in a single step..." << std::endl;

    struct ggml_init_params prefill_params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * prefill_ctx = ggml_init(prefill_params);
    if (!prefill_ctx) {
        std::cerr << "Failed to initialize prefill context!" << std::endl;
        compressor.stop();
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_tensor * input_tokens_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, L);
    ggml_set_input(input_tokens_prefill);
    struct ggml_tensor * positions_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, L);
    ggml_set_input(positions_prefill);
    struct ggml_tensor * mask_prefill = ggml_new_tensor_2d(prefill_ctx, GGML_TYPE_F16, L, L);
    ggml_set_input(mask_prefill);

    struct ggml_tensor * prefill_logits = nullptr;
    std::vector<struct ggml_tensor *> prefill_k_layers(n_layers, nullptr);
    std::vector<struct ggml_tensor *> prefill_v_layers(n_layers, nullptr);
    struct ggml_cgraph * prefill_graph = build_prefill_graph(
        prefill_ctx, model, input_tokens_prefill, positions_prefill, mask_prefill,
        &prefill_logits, &prefill_k_layers, &prefill_v_layers
    );
    ggml_set_output(prefill_logits);

    ggml_gallocr_t prefill_galloc = ggml_gallocr_new(buft);
    if (!prefill_galloc || !ggml_gallocr_alloc_graph(prefill_galloc, prefill_graph)) {
        std::cerr << "Failed to allocate memory for prefill graph!" << std::endl;
        ggml_free(prefill_ctx);
        compressor.stop();
        ggml_backend_free(backend);
        return 1;
    }

    // Set inputs for prefill
    ggml_backend_tensor_set(input_tokens_prefill, prompt_tokens.data(), 0, L * sizeof(int32_t));

    std::vector<int32_t> pos_host(L);
    for (int i = 0; i < L; ++i) pos_host[i] = i;
    ggml_backend_tensor_set(positions_prefill, pos_host.data(), 0, L * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask_host(L * L, ggml_fp32_to_fp16(0.0f));
    for (int i = 0; i < L; ++i) {
        for (int j = i + 1; j < L; ++j) {
            mask_host[i * L + j] = ggml_fp32_to_fp16(-INFINITY);
        }
    }
    ggml_backend_tensor_set(mask_prefill, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, prefill_graph) != GGML_STATUS_SUCCESS) {
        std::cerr << "Error: Prefill graph compute failed!" << std::endl;
        ggml_gallocr_free(prefill_galloc);
        ggml_free(prefill_ctx);
        compressor.stop();
        ggml_backend_free(backend);
        return 1;
    }

    // Retrieve raw K/V activations
    std::vector<std::vector<float>> k_activations(n_layers, std::vector<float>(L * F_test));
    std::vector<std::vector<float>> v_activations(n_layers, std::vector<float>(L * F_test));
    for (int l = 0; l < n_layers; ++l) {
        ggml_backend_tensor_get(prefill_k_layers[l], k_activations[l].data(), 0, L * F_test * sizeof(float));
        ggml_backend_tensor_get(prefill_v_layers[l], v_activations[l].data(), 0, L * F_test * sizeof(float));
    }

    // Segment activations into blocks and enqueue SVD compression
    int num_blocks = (L + 63) / 64;
    std::vector<std::vector<float>> active_k_dense(n_layers, std::vector<float>(64 * F_test, 0.0f));
    std::vector<std::vector<float>> active_v_dense(n_layers, std::vector<float>(64 * F_test, 0.0f));
    std::map<int, std::vector<float>> persistent_k_dense;
    std::map<int, std::vector<float>> persistent_v_dense;
    std::vector<std::vector<int32_t>> seq_lens_by_layer(n_layers, std::vector<int32_t>(n_slots, 0));

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int block_start = block_idx * 64;
        int block_len = std::min(64, L - block_start);

        for (int l = 0; l < n_layers; ++l) {
            auto & engine = *kv_engines[l];
            float* k_block = &k_activations[l][block_start * F_test];
            float* v_block = &v_activations[l][block_start * F_test];

            // 1. Set Anchor (first token of block)
            std::vector<ggml_fp16_t> k_fp16(F_test);
            std::vector<ggml_fp16_t> v_fp16(F_test);
            for (int i = 0; i < F_test; ++i) {
                k_fp16[i] = ggml_fp32_to_fp16(k_block[i]);
                v_fp16[i] = ggml_fp32_to_fp16(v_block[i]);
            }
            ggml_backend_tensor_set(engine.get_anchors_K(), k_fp16.data(), block_idx * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(engine.get_anchors_V(), v_fp16.data(), block_idx * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));

            // Record the actual sequence position of this block's anchor token
            int32_t anchor_pos = block_start;
            ggml_backend_tensor_set(engine.get_anchor_positions(), &anchor_pos, block_idx * sizeof(int32_t), sizeof(int32_t));

            engine.get_state_table().transition(block_idx, BlockState::Freed, BlockState::DenseResident);

            // 2. Set deltas relative to anchor and submit to compressor
            if (block_len == 64) {
                for (int t = 1; t < block_len; ++t) {
                    int offset = (t - 1) * F_test;
                    for (int i = 0; i < F_test; ++i) {
                        active_k_dense[l][offset + i] = k_block[t * F_test + i] - k_block[i];
                        active_v_dense[l][offset + i] = v_block[t * F_test + i] - v_block[i];
                    }
                }

                engine.get_state_table().transition(block_idx, BlockState::DenseResident, BlockState::Compressing);

                int p_key = l * 10000 + block_idx;
                persistent_k_dense[p_key] = active_k_dense[l];
                persistent_v_dense[p_key] = active_v_dense[l];

                CompressJob job;
                job.session_id = 42;
                job.block_id = block_idx;
                job.block_size = block_len - 1;
                job.feat_dim = F_test;
                job.rank = rank;
                job.dense_k_ptr = persistent_k_dense[p_key].data();
                job.dense_v_ptr = persistent_v_dense[p_key].data();

                job.out_u_ptr = reinterpret_cast<int8_t*>(engine.get_U()->data) + block_idx * 64 * rank;
                job.out_u_scale = reinterpret_cast<ggml_fp16_t*>(engine.get_U_scale()->data) + block_idx;
                job.out_vk_ptr = reinterpret_cast<ggml_fp16_t*>(engine.get_VK()->data) + block_idx * rank * F_test;
                job.out_vv_ptr = reinterpret_cast<ggml_fp16_t*>(engine.get_VV()->data) + block_idx * rank * F_test;
                job.out_scale = reinterpret_cast<ggml_fp16_t*>(engine.get_scales()->data) + block_idx;
                job.state_table = &engine.get_state_table();

                compressor.submit(job);

                seq_lens_by_layer[l][block_idx] = 63;
            } else {
                if (block_len > 1) {
                    for (int t = 1; t < block_len; ++t) {
                        int offset = (t - 1) * F_test;
                        for (int i = 0; i < F_test; ++i) {
                            active_k_dense[l][offset + i] = k_block[t * F_test + i] - k_block[i];
                            active_v_dense[l][offset + i] = v_block[t * F_test + i] - v_block[i];
                        }
                    }
                    int p_key = l * 10000 + block_idx;
                    persistent_k_dense[p_key] = active_k_dense[l];
                    persistent_v_dense[p_key] = active_v_dense[l];
                }
                seq_lens_by_layer[l][block_idx] = 0;
            }
            ggml_backend_tensor_set(engine.get_seq_lens(), seq_lens_by_layer[l].data(), 0, seq_lens_by_layer[l].size() * sizeof(int32_t));

            // 3. Compute Mean Key Descriptor for Layer 0
            if (l == 0) {
                std::vector<float> avg_k(F_test, 0.0f);
                for (int i = 0; i < F_test; ++i) {
                    for (int t = 0; t < block_len; ++t) {
                        avg_k[i] += k_block[t * F_test + i];
                    }
                    avg_k[i] /= block_len;
                }

                std::vector<float> desc(desc_dim, 0.0f);
                for (int r = 0; r < desc_dim; ++r) {
                    float sum = 0.0f;
                    for (int c = 0; c < head_dim; ++c) {
                        sum += avg_k[c] * W_proj_host[r * head_dim + c];
                    }
                    desc[r] = sum;
                }
                float sum_sq = 0.0f;
                for (float val : desc) sum_sq += val * val;
                float norm = std::sqrt(sum_sq) + 1e-8f;
                for (float & val : desc) val /= norm;
                ggml_backend_tensor_set(engine.get_desc_matrix(), desc.data(), block_idx * desc_dim * sizeof(float), desc_dim * sizeof(float));
            }
        }
    }

    // Wait for SVD compressions to catch up
    std::cout << "[DiffKV Native] Waiting for background SVD compressor to catch up..." << std::endl;
    for (int s = 0; s < num_blocks; ++s) {
        auto start_time = std::chrono::steady_clock::now();
        while (true) {
            bool all_done = true;
            for (int l = 0; l < n_layers; ++l) {
                BlockState st = kv_engines[l]->get_state_table().get(s);
                if (st != BlockState::CompressedResident && st != BlockState::DenseResident) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time
            ).count();
            if (elapsed > 4000) {
                std::cerr << "Warning: Timeout waiting for compression of slot " << s << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // Sample the next token from prefill logits (greedy argmax on last token)
    std::vector<float> prefill_output_logits(n_vocab);
    ggml_backend_tensor_get(prefill_logits, prefill_output_logits.data(), (L - 1) * n_vocab * sizeof(float), n_vocab * sizeof(float));

    std::vector<std::pair<float, int>> prefill_top_k;
    for (int i = 0; i < n_vocab; ++i) {
        prefill_top_k.push_back({prefill_output_logits[i], i});
    }
    std::sort(prefill_top_k.begin(), prefill_top_k.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
    });

    std::cout << "\n[Prefill Phase Top predictions]:\n";
    for (int k = 0; k < 5; ++k) {
        std::cout << "  " << k << ": \"" << model.token_to_piece(prefill_top_k[k].second) << "\" (id: " << prefill_top_k[k].second << ", logit: " << prefill_top_k[k].first << ")\n";
    }

    int32_t first_decode_token = prefill_top_k[0].second;

    // Clean up prefill graph memory
    ggml_gallocr_free(prefill_galloc);
    ggml_free(prefill_ctx);

    // ── 2. DECODE PHASE (Autoregressive sparse attention) ──
    std::cout << "[DiffKV Native] Preparing Decode Graph..." << std::endl;

    struct ggml_init_params decode_params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * decode_ctx = ggml_init(decode_params);
    if (!decode_ctx) {
        std::cerr << "Failed to initialize decode context!" << std::endl;
        compressor.stop();
        return 1;
    }

    struct ggml_tensor * input_token_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
    ggml_set_input(input_token_decode);
    struct ggml_tensor * position_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_I32, 1);
    ggml_set_input(position_decode);
    struct ggml_tensor * W_proj_decode = ggml_new_tensor_2d(decode_ctx, GGML_TYPE_F32, head_dim, desc_dim);
    ggml_set_input(W_proj_decode);
    struct ggml_tensor * slots_mask_decode = ggml_new_tensor_1d(decode_ctx, GGML_TYPE_F32, n_slots);
    ggml_set_input(slots_mask_decode);

    // Initialize userdata for decode
    std::vector<diffkv::CustomAttnUserData> userdata(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        userdata[l].kv_engine = kv_engines[l].get();
        userdata[l].slot_indices = nullptr;
        userdata[l].n_q_heads = model.get_config().n_head;
        userdata[l].n_kv_heads = model.get_config().n_head_kv;
        userdata[l].rank = rank;
        userdata[l].S_max = 64;
        userdata[l].K = std::min(8, num_blocks);
        userdata[l].D = head_dim;
        userdata[l].scale = 1.0f / std::sqrt((float)head_dim);
        userdata[l].has_rope = true;
        userdata[l].rope_freq_base = model.get_config().rope_freq_base;
    }

    struct ggml_tensor * decode_logits = nullptr;
    struct ggml_tensor * selected_slots = nullptr;
    std::vector<struct ggml_tensor *> decode_k_layers(n_layers, nullptr);
    std::vector<struct ggml_tensor *> decode_v_layers(n_layers, nullptr);

    struct ggml_cgraph * decode_graph = build_decode_graph(
        decode_ctx, model, input_token_decode, position_decode, W_proj_decode,
        kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
        slots_mask_decode,
        userdata.data(), &decode_logits, &selected_slots,
        &decode_k_layers, &decode_v_layers
    );
    ggml_set_output(decode_logits);
    if (selected_slots) ggml_set_output(selected_slots);

    ggml_gallocr_t decode_galloc = ggml_gallocr_new(buft);
    if (!decode_galloc || !ggml_gallocr_alloc_graph(decode_galloc, decode_graph)) {
        std::cerr << "Failed to allocate memory for decode graph!" << std::endl;
        ggml_free(decode_ctx);
        compressor.stop();
        return 1;
    }

    // Upload W_proj
    ggml_backend_tensor_set(W_proj_decode, W_proj_host.data(), 0, W_proj_host.size() * sizeof(float));

    // Start autoregressive loop
    std::cout << "[DiffKV Native] Starting Autoregressive Generation..." << std::endl;
    std::cout << "[Response] " << std::flush;

    int32_t last_token = first_decode_token;
    std::string first_piece = model.token_to_piece(last_token);
    std::cout << first_piece << std::flush;

    std::vector<int32_t> generated_tokens;
    generated_tokens.push_back(last_token);
    const int max_generate = 50;

    int active_slot = (L - 1) / 64;
    int active_block_tokens = L - active_slot * 64;
    if (active_block_tokens == 64) {
        active_slot++;
        active_block_tokens = 0;
    }

    // Restore active delta dense buffers if continuing a block
    for (int l = 0; l < n_layers; ++l) {
        int p_key = l * 10000 + active_slot;
        if (active_block_tokens > 0 && persistent_k_dense.count(p_key)) {
            active_k_dense[l] = persistent_k_dense[p_key];
            active_v_dense[l] = persistent_v_dense[p_key];
        } else {
            std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
            std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
        }
    }

    for (int step = 0; step < max_generate; ++step) {
        int current_pos = L + step;

        ggml_backend_tensor_set(input_token_decode, &last_token, 0, sizeof(last_token));
        ggml_backend_tensor_set(position_decode, &current_pos, 0, sizeof(current_pos));

        // Set slots mask: 0.0f for occupied compressed slots (<= active_slot - 1), -1e10f for unoccupied/dense slots
        std::vector<float> slots_mask_host(n_slots, -1e10f);
        int occupied_up_to = active_slot - 1;
        for (int i = 0; i <= occupied_up_to; ++i) {
            if (i >= 0 && i < n_slots) {
                slots_mask_host[i] = 0.0f;
            }
        }
        ggml_backend_tensor_set(slots_mask_decode, slots_mask_host.data(), 0, n_slots * sizeof(float));

        // Update K and dense active window parameters in userdata dynamically
        int current_k = std::max(0, std::min(8, active_slot));
        for (int l = 0; l < n_layers; ++l) {
            userdata[l].K = current_k;
            userdata[l].active_k_dense = active_k_dense[l].data();
            userdata[l].active_v_dense = active_v_dense[l].data();
            userdata[l].active_block_tokens = active_block_tokens;
            userdata[l].active_slot = active_slot;
        }

        if (ggml_backend_graph_compute(backend, decode_graph) != GGML_STATUS_SUCCESS) {
            std::cerr << "Error: Decode graph compute failed at step " << step << std::endl;
            break;
        }

        // (slot selection debug logging removed)

        // Sample next token greedy
        std::vector<float> output_logits(n_vocab);
        ggml_backend_tensor_get(decode_logits, output_logits.data(), 0, n_vocab * sizeof(float));

        // Apply repetition penalty (1.15) — penalise tokens already generated
        constexpr float REP_PENALTY = 1.15f;
        for (int32_t tok : generated_tokens) {
            if (tok >= 0 && tok < n_vocab) {
                float& l = output_logits[tok];
                l = (l > 0.0f) ? l / REP_PENALTY : l * REP_PENALTY;
            }
        }
        // Also penalise the current input token
        if (last_token >= 0 && last_token < n_vocab) {
            float& l = output_logits[last_token];
            l = (l > 0.0f) ? l / REP_PENALTY : l * REP_PENALTY;
        }

        // Print top-5 predictions for debugging
        struct LogitPair {
            int id;
            float val;
        };
        std::vector<LogitPair> logits_sorted(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            logits_sorted[i] = { i, output_logits[i] };
        }
        std::sort(logits_sorted.begin(), logits_sorted.end(), [](const LogitPair& a, const LogitPair& b) {
            return a.val > b.val;
        });
        std::cout << "\n[Step " << step << " Top predictions]:\n";
        for (int i = 0; i < std::min(5, n_vocab); ++i) {
            std::cout << "  " << i << ": \"" << model.token_to_piece(logits_sorted[i].id) << "\" (id: " << logits_sorted[i].id << ", logit: " << logits_sorted[i].val << ")\n";
        }

        // Greedy argmax
        int32_t next_token = logits_sorted[0].id;
        float best_logit = logits_sorted[0].val;

        if (model.is_eog_token(next_token) || next_token == model.token_eos()) {
            std::cout << " [EOS]" << std::endl;
            break;
        }

        std::string piece = model.token_to_piece(next_token);
        std::cout << piece << std::flush;

        generated_tokens.push_back(next_token);
        last_token = next_token;

        // Ingest next token's projected keys and values into the block pool for all layers
        for (int l = 0; l < n_layers; ++l) {
            auto & engine = *kv_engines[l];
            std::vector<float> k_host(F_test);
            std::vector<float> v_host(F_test);
            ggml_backend_tensor_get(decode_k_layers[l], k_host.data(), 0, F_test * sizeof(float));
            ggml_backend_tensor_get(decode_v_layers[l], v_host.data(), 0, F_test * sizeof(float));

            if (active_block_tokens == 0) {
                // First token of new block: write as anchor
                std::vector<ggml_fp16_t> k_fp16(F_test);
                std::vector<ggml_fp16_t> v_fp16(F_test);
                for (int i = 0; i < F_test; ++i) {
                    k_fp16[i] = ggml_fp32_to_fp16(k_host[i]);
                    v_fp16[i] = ggml_fp32_to_fp16(v_host[i]);
                }
                ggml_backend_tensor_set(engine.get_anchors_K(), k_fp16.data(), active_slot * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(engine.get_anchors_V(), v_fp16.data(), active_slot * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                // Record actual sequence position of this decode block's anchor token (only once, from layer 0)
                if (l == 0) {
                    int32_t anchor_seq_pos = current_pos;
                    for (int ll = 0; ll < n_layers; ++ll) {
                        ggml_backend_tensor_set(kv_engines[ll]->get_anchor_positions(), &anchor_seq_pos, active_slot * sizeof(int32_t), sizeof(int32_t));
                    }
                }
                engine.get_state_table().transition(active_slot, BlockState::Freed, BlockState::DenseResident);
            } else {
                // Subsequent token: compute delta relative to anchor.
                // The block may be DenseResident (not yet compressed) or CompressedResident
                // (if it was compressed from a previous decode step). Only transition if compressed.
                BlockState cur_state = engine.get_state_table().get(active_slot);
                if (cur_state == BlockState::CompressedResident) {
                    engine.get_state_table().transition(active_slot, BlockState::CompressedResident, BlockState::DenseResident);
                }
                // (if DenseResident, no transition needed — we can write directly)

                std::vector<ggml_fp16_t> anchor_k_fp16(F_test);
                ggml_backend_tensor_get(engine.get_anchors_K(), anchor_k_fp16.data(), active_slot * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                std::vector<ggml_fp16_t> anchor_v_fp16(F_test);
                ggml_backend_tensor_get(engine.get_anchors_V(), anchor_v_fp16.data(), active_slot * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));

                int offset = (active_block_tokens - 1) * F_test;
                for (int i = 0; i < F_test; ++i) {
                    active_k_dense[l][offset + i] = k_host[i] - ggml_fp16_to_fp32(anchor_k_fp16[i]);
                    active_v_dense[l][offset + i] = v_host[i] - ggml_fp16_to_fp32(anchor_v_fp16[i]);
                }
            }
        }

        active_block_tokens++;
        if (active_block_tokens == 64) {
            // Trigger synchronous SVD compression for the completed block before moving to the next slot
            for (int l = 0; l < n_layers; ++l) {
                auto & engine = *kv_engines[l];
                engine.get_state_table().transition(active_slot, BlockState::DenseResident, BlockState::Compressing);

                CompressJob job;
                job.session_id = 42;
                job.block_id = active_slot;
                job.block_size = 63; // active_block_tokens - 1
                job.feat_dim = F_test;
                job.rank = rank;
                job.dense_k_ptr = active_k_dense[l].data();
                job.dense_v_ptr = active_v_dense[l].data();

                job.out_u_ptr = reinterpret_cast<int8_t*>(engine.get_U()->data) + active_slot * 64 * rank;
                job.out_u_scale = reinterpret_cast<ggml_fp16_t*>(engine.get_U_scale()->data) + active_slot;
                job.out_vk_ptr = reinterpret_cast<ggml_fp16_t*>(engine.get_VK()->data) + active_slot * rank * F_test;
                job.out_vv_ptr = reinterpret_cast<ggml_fp16_t*>(engine.get_VV()->data) + active_slot * rank * F_test;
                job.out_scale = reinterpret_cast<ggml_fp16_t*>(engine.get_scales()->data) + active_slot;
                job.state_table = &engine.get_state_table();

                compressor.compress_sync(job);

                seq_lens_by_layer[l][active_slot] = 63;
                ggml_backend_tensor_set(engine.get_seq_lens(), seq_lens_by_layer[l].data(), 0, seq_lens_by_layer[l].size() * sizeof(int32_t));

                if (l == 0) {
                    // Update mean key descriptor for Layer 0
                    std::vector<float> avg_k(head_dim, 0.0f);
                    std::vector<ggml_fp16_t> anchor_k_fp16(F_test);
                    ggml_backend_tensor_get(engine.get_anchors_K(), anchor_k_fp16.data(), active_slot * F_test * sizeof(ggml_fp16_t), F_test * sizeof(ggml_fp16_t));
                    for (int i = 0; i < head_dim; ++i) {
                        avg_k[i] += ggml_fp16_to_fp32(anchor_k_fp16[i]);
                        for (int t = 0; t < 63; ++t) {
                            avg_k[i] += active_k_dense[0][t * F_test + i];
                        }
                        avg_k[i] /= 64;
                    }

                    std::vector<float> desc(desc_dim, 0.0f);
                    for (int r = 0; r < desc_dim; ++r) {
                        float sum = 0.0f;
                        for (int c = 0; c < head_dim; ++c) {
                            sum += avg_k[c] * W_proj_host[r * head_dim + c];
                        }
                        desc[r] = sum;
                    }
                    float sum_sq = 0.0f;
                    for (float val : desc) sum_sq += val * val;
                    float norm = std::sqrt(sum_sq) + 1e-8f;
                    for (float & val : desc) val /= norm;
                    ggml_backend_tensor_set(engine.get_desc_matrix(), desc.data(), active_slot * desc_dim * sizeof(float), desc_dim * sizeof(float));
                }
            }

            active_slot++;
            active_block_tokens = 0;
            // Clear active buffers for next block
            for (int l = 0; l < n_layers; ++l) {
                std::fill(active_k_dense[l].begin(), active_k_dense[l].end(), 0.0f);
                std::fill(active_v_dense[l].begin(), active_v_dense[l].end(), 0.0f);
            }
        }
    }
    std::cout << std::endl;

    // Stop compressor and cleanup
    compressor.stop();

    ggml_gallocr_free(decode_galloc);
    ggml_free(decode_ctx);
    ggml_backend_free(backend);

    std::cout << "[DiffKV Native] Text generation completed successfully!" << std::endl;
    return 0;
}
