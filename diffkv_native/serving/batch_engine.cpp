#include "serving/batch_engine.hpp"
#include "runtime/diffkv_model.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../third_party/llama.cpp/ggml/src/ggml-impl.h"
#include "runtime/native_block_pool.hpp"
#include "native_core/srl/diffkv_srl.hpp"
#include "native_core/srl/factual_alignment.hpp"
#include "native_core/srl/query_router.hpp"
#include "runtime/diffkv_attention.hpp"
#include "native_core/compression/async_compressor.hpp"
#include "native_core/kv_runtime_manager.hpp"
#include "native_core/srl/factual_store.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>

namespace diffkv {

// k_rotated: output buffer, same shape.
static void apply_rope_neox_cpu_fast(
    const float* k_raw, float* k_rotated,
    const float* cos_table, const float* sin_table,
    int num_tokens, int kv_heads, int head_dim
) {
    const int D    = head_dim;
    const int half = D / 2;
    for (int t = 0; t < num_tokens; ++t) {
        const float* cos_t = cos_table + t * half;
        const float* sin_t = sin_table + t * half;
        for (int h = 0; h < kv_heads; ++h) {
            const float* src = k_raw     + (t * kv_heads + h) * D;
            float*       dst = k_rotated + (t * kv_heads + h) * D;
            for (int i = 0; i < half; ++i) {
                float cos_th = cos_t[i];
                float sin_th = sin_t[i];
                float x      = src[i];
                float y      = src[i + half];
                dst[i]        = x * cos_th - y * sin_th;
                dst[i + half] = y * cos_th + x * sin_th;
            }
        }
    }
}

// Helper to build the Qwen 2.5 dense prefill graph using causal flash attention
static struct ggml_cgraph * build_prefill_graph(
    struct ggml_context * ctx,
    DiffKVModel & model,
    struct ggml_tensor * input_tokens,
    struct ggml_tensor * positions,
    struct ggml_tensor * mask,
    std::vector<struct ggml_tensor *> * prior_k_ctx,
    std::vector<struct ggml_tensor *> * prior_v_ctx,
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

        // 6. Concatenate prior context with current chunk along seq dim (dim=1 in permuted layout)
        struct ggml_tensor * k_ctx_perm = k_perm;
        struct ggml_tensor * v_ctx_perm = v_perm;
        bool has_prior = (prior_k_ctx && (*prior_k_ctx)[l] != nullptr);
        if (has_prior) {
            struct ggml_tensor * pk = ggml_permute(ctx, (*prior_k_ctx)[l], 0, 2, 1, 3);
            struct ggml_tensor * pv = ggml_permute(ctx, (*prior_v_ctx)[l], 0, 2, 1, 3);
            k_ctx_perm = ggml_concat(ctx, pk, k_perm, 1);
            v_ctx_perm = ggml_concat(ctx, pv, v_perm, 1);
        }

        // 7. Flash Attention
        float scale_val = 1.0f / std::sqrt((float)head_dim);
        struct ggml_tensor * attn_out_perm = ggml_flash_attn_ext(ctx, q_perm, k_ctx_perm, v_ctx_perm, mask, scale_val, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out_perm, GGML_PREC_F32);

        // 8. Flatten back to [n_embd, q_len]
        struct ggml_tensor * attn_out = ggml_reshape_2d(ctx, attn_out_perm, config.n_embd, q->ne[1]);

        // 9. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 10. Residual connection
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
static struct ggml_cgraph * build_decode_graph(
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
    struct ggml_tensor ** out_concat_k = nullptr,
    struct ggml_tensor ** out_concat_v = nullptr
) {
    const auto & config = model.get_config();
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    int F_test = config.n_embd / config.n_head * config.n_head_kv;
    struct ggml_tensor * concat_k = nullptr;
    struct ggml_tensor * concat_v = nullptr;
    if (out_concat_k) {
        concat_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F_test, config.n_layer);
        *out_concat_k = concat_k;
    }
    if (out_concat_v) {
        concat_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F_test, config.n_layer);
        *out_concat_v = concat_v;
    }

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
        if (concat_k) {
            struct ggml_tensor * k_view = ggml_view_1d(ctx, concat_k, F_test, l * F_test * sizeof(float));
            struct ggml_tensor * copy_k = ggml_cpy(ctx, k, k_view);
            ggml_build_forward_expand(gf, copy_k);
        }
        if (concat_v) {
            struct ggml_tensor * v_view = ggml_view_1d(ctx, concat_v, F_test, l * F_test * sizeof(float));
            struct ggml_tensor * copy_v = ggml_cpy(ctx, v, v_view);
            ggml_build_forward_expand(gf, copy_v);
        }

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

        // 5. Custom Metal Attention (wrapping forward/sparse Metal kernel)
        struct ggml_tensor * attn_out = nullptr;
        if (userdata && selected_slots) {
            struct ggml_tensor * kv_concat = ggml_concat(ctx, k, v, 0);
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
        
        // 6. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out);
        if (layer.bo) attn_proj = ggml_add(ctx, attn_proj, layer.bo);

        // 7. Residual connection
        cur = ggml_add(ctx, cur, attn_proj);

        // 8. FFN RMSNorm
        h = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
        h = ggml_mul(ctx, h, layer.ffn_norm);

        // 9. FFN SwiGLU
        struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, h);
        struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.ffn_up, h);

        struct ggml_tensor * gate_silu = ggml_silu(ctx, gate);
        struct ggml_tensor * ffn_out   = ggml_mul(ctx, gate_silu, up);
        struct ggml_tensor * ffn_proj  = ggml_mul_mat(ctx, layer.ffn_down, ffn_out);

        // 10. Residual connection
        cur = ggml_add(ctx, cur, ffn_proj);
    }

    // 11. Final RMSNorm
    struct ggml_tensor * final_norm = ggml_rms_norm(ctx, cur, config.rms_norm_eps);
    final_norm = ggml_mul(ctx, final_norm, model.get_output_norm());

    // 12. Multiply final_norm [n_embd, 1] by model.get_output() [n_embd, n_vocab] -> logits [n_vocab, 1]
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.get_output(), final_norm);
    *out_logits = logits;

    ggml_build_forward_expand(gf, logits);

    if (out_concat_k && *out_concat_k) {
        struct ggml_tensor * dummy_concat_k = ggml_view_1d(ctx, *out_concat_k, (*out_concat_k)->ne[0] * (*out_concat_k)->ne[1], 0);
        ggml_build_forward_expand(gf, dummy_concat_k);
    }
    if (out_concat_v && *out_concat_v) {
        struct ggml_tensor * dummy_concat_v = ggml_view_1d(ctx, *out_concat_v, (*out_concat_v)->ne[0] * (*out_concat_v)->ne[1], 0);
        ggml_build_forward_expand(gf, dummy_concat_v);
    }

    return gf;
}

// Thread-safe sampling implementation supporting Repetition Penalty, Temperature, and Top-P
static int32_t sample_token(
    std::vector<float>& logits,
    float temperature,
    float top_p,
    float repetition_penalty,
    const std::vector<int32_t>& penalty_tokens
) {
    int n_vocab = logits.size();
    
    // 1. Repetition penalty
    if (repetition_penalty != 1.0f) {
        std::unordered_set<int32_t> unique_penalized(penalty_tokens.begin(), penalty_tokens.end());
        for (int32_t tok : unique_penalized) {
            if (tok >= 0 && tok < n_vocab) {
                float& val = logits[tok];
                val = (val > 0.0f) ? val / repetition_penalty : val * repetition_penalty;
            }
        }
    }
    
    // 2. Greedy sampling (temperature <= 0.01)
    if (temperature <= 0.01f) {
        float max_val = -INFINITY;
        int32_t best_tok = 0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                best_tok = i;
            }
        }
        return best_tok;
    }
    
    // 3. Temperature scaling
    std::vector<double> probs(n_vocab);
    double max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        logits[i] /= temperature;
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    // Softmax
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] /= sum;
    }
    
    // 4. Top-P sampling
    if (top_p < 1.0f) {
        std::vector<std::pair<double, int32_t>> prob_indices(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            prob_indices[i] = {probs[i], i};
        }
        std::sort(prob_indices.begin(), prob_indices.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        
        double cum_sum = 0.0;
        size_t keep_count = 0;
        for (size_t i = 0; i < prob_indices.size(); ++i) {
            cum_sum += prob_indices[i].first;
            keep_count++;
            if (cum_sum >= top_p) {
                break;
            }
        }
        
        // Re-normalize top-p subset
        double subset_sum = 0.0;
        for (size_t i = 0; i < keep_count; ++i) {
            subset_sum += prob_indices[i].first;
        }
        
        std::vector<double> norm_probs(keep_count);
        for (size_t i = 0; i < keep_count; ++i) {
            norm_probs[i] = prob_indices[i].first / subset_sum;
        }
        
        // Random sample
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double r = dis(gen);
        
        double current_r = 0.0;
        for (size_t i = 0; i < keep_count; ++i) {
            current_r += norm_probs[i];
            if (r <= current_r) {
                return prob_indices[i].second;
            }
        }
        return prob_indices.back().second;
    } else {
        // Standard multinomial sample
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double r = dis(gen);
        
        double current_r = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            current_r += probs[i];
            if (r <= current_r) {
                return i;
            }
        }
        return n_vocab - 1;
    }
}

DiffKVBatchEngine::DiffKVBatchEngine(
    DiffKVModel* model,
    ggml_backend_t backend,
    ggml_backend_sched_t sched,
    KVRuntimeManager* runtime_manager,
    ProductionSessionManager* session_manager
) : model_(model),
    backend_(backend),
    sched_(sched),
    runtime_manager_(runtime_manager),
    session_manager_(session_manager) {
    
    // Initialize stop word patterns for the inverted index to filter out common tokens
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
        auto t = model_->tokenize(word, false);
        for (int32_t tok : t) {
            stop_token_ids_.insert(tok);
        }
    }
    for (int i = 0; i < 200; ++i) {
        stop_token_ids_.insert(i);
    }
    runtime_manager_->get_ingest_manager().set_stop_token_ids(&stop_token_ids_);
}

DiffKVBatchEngine::~DiffKVBatchEngine() {
    stop();
}

void DiffKVBatchEngine::start() {
    if (running_.exchange(true)) return;
    worker_thread_ = std::thread(&DiffKVBatchEngine::run_loop, this);
}

void DiffKVBatchEngine::stop() {
    if (!running_.exchange(false)) return;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

std::shared_ptr<BatchRequest> DiffKVBatchEngine::submit(
    const std::string& session_id,
    const std::string& prompt,
    int max_tokens,
    float temperature,
    float top_p,
    float repetition_penalty
) {
    auto req = std::make_shared<BatchRequest>(
        session_id,
        prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty
    );
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_.push(req);
        queue_cv_.notify_one();
    }
    
    return req;
}

void DiffKVBatchEngine::cancel(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Cancel in queue
    std::queue<std::shared_ptr<BatchRequest>> temp_q;
    while (!queue_.empty()) {
        auto req = queue_.front();
        queue_.pop();
        if (req->session_id == session_id) {
            req->cancelled = true;
            req->finish_stream();
        } else {
            temp_q.push(req);
        }
    }
    queue_ = std::move(temp_q);
}

void DiffKVBatchEngine::run_loop() {
    while (running_) {
        std::shared_ptr<BatchRequest> req;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !queue_.empty() || !running_;
            });
            
            if (!running_) break;
            
            req = queue_.front();
            queue_.pop();
        }
        
        if (req && !req->cancelled) {
            try {
                process_request(req);
            } catch (const std::exception& e) {
                std::cerr << "[DiffKV BatchEngine] Exception during request execution: " << e.what() << std::endl;
                req->set_error(std::string("Internal batch engine exception: ") + e.what());
            } catch (...) {
                std::cerr << "[DiffKV BatchEngine] Unknown exception during request execution" << std::endl;
                req->set_error("Unknown internal batch engine exception");
            }
        }
    }
}

void DiffKVBatchEngine::process_request(const std::shared_ptr<BatchRequest>& req) {
    if (!session_manager_) {
        req->set_error("Session manager is not configured.");
        return;
    }

    auto session = session_manager_->get_session(req->session_id);
    if (!session) {
        req->set_error("Failed to load or create session " + req->session_id);
        return;
    }
    
    // Re-verify residency and register as active
    session_manager_->ensure_residency(req->session_id);
    runtime_manager_->get_ingest_manager().set_session_id(req->session_id);
    
    int n_vocab = model_->get_config().n_vocab;
    int n_layers = model_->get_config().n_layer;
    int head_dim = model_->get_config().n_embd / model_->get_config().n_head;
    int kv_heads = model_->get_config().n_head_kv;
    int desc_dim = 64;
    int n_slots = model_->get_config().n_ctx / 64;
    
    // Override max context slots from env if present
    if (const char* env_slots = std::getenv("DIFFKV_MAX_CONTEXT_SLOTS")) {
        n_slots = std::stoi(env_slots);
    }
    
    int micro_block_size = 64;
    if (const char* env_mbs = std::getenv("DIFFKV_MICRO_BLOCK_SIZE")) {
        micro_block_size = std::stoi(env_mbs);
    }
    
    // Hyperparameters for SRL candidate selection
    int srl_k_semantic = 32;
    int srl_k_lexical = 8;
    int srl_k_graph = 8;
    int srl_k_recency = 8;
    int srl_k_keep = 16;
    
    if (const char* env = std::getenv("DIFFKV_SRL_K_SEM")) srl_k_semantic = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_LEX")) srl_k_lexical = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_GRAPH")) srl_k_graph = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_RECENCY")) srl_k_recency = std::stoi(env);
    if (const char* env = std::getenv("DIFFKV_SRL_K_KEEP")) srl_k_keep = std::stoi(env);
    
    srl_k_semantic = std::min(srl_k_semantic, n_slots);
    srl_k_keep = std::min(srl_k_keep, n_slots);
    
    int srl_k_host = 1 + srl_k_recency + srl_k_lexical + srl_k_semantic + srl_k_graph;
    int F_test = kv_heads * head_dim;
    
    // Initialize W_proj_host
    std::vector<float> W_proj_host(head_dim * desc_dim);
    {
        std::mt19937 rand_gen(42);
        std::normal_distribution<float> rand_dist(0.0f, 1.0f);
        for (int r = 0; r < desc_dim; ++r) {
            float sum_sq = 0.0f;
            for (int c = 0; c < head_dim; ++c) {
                float val = rand_dist(rand_gen);
                W_proj_host[r * head_dim + c] = val;
                sum_sq += val * val;
            }
            float norm = std::sqrt(sum_sq) + 1e-8f;
            for (int c = 0; c < head_dim; ++c) {
                W_proj_host[r * head_dim + c] /= norm;
            }
        }
    }
    
    // Initialize session-specific lists if empty
    if (session->active_k_dense.empty()) {
        session->active_k_dense.assign(n_layers, AlignedFloatVector(16384 * F_test, 0.0f));
        session->active_v_dense.assign(n_layers, AlignedFloatVector(16384 * F_test, 0.0f));
        session->active_positions_dense.assign(16384, 0);
    }
    if (session->seq_lens_by_layer.empty()) {
        session->seq_lens_by_layer.assign(n_layers, std::vector<int32_t>(n_slots, 0));
    }

    int total_positions = 0;
    auto sync_active_dense_buffers = [&](
        std::vector<int>& dense_start_positions,
        std::vector<int>& total_dense_tokens
    ) {
        for (int l = 0; l < n_layers; ++l) {
            auto & b_list = runtime_manager_->get_ingest_manager().get_blocks(l);
            int curr_token_idx = 0;
            bool found_first = false;

            std::fill(session->active_k_dense[l].begin(), session->active_k_dense[l].end(), 0.0f);
            std::fill(session->active_v_dense[l].begin(), session->active_v_dense[l].end(), 0.0f);

            for (auto & block : b_list) {
                if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                    if (!found_first) {
                        dense_start_positions[l] = block->anchor_idx;
                        found_first = true;
                    }

                    std::memcpy(
                        session->active_k_dense[l].data() + curr_token_idx * F_test,
                        block->anchor_k.data(),
                        F_test * sizeof(float)
                    );
                    std::memcpy(
                        session->active_v_dense[l].data() + curr_token_idx * F_test,
                        block->anchor_v.data(),
                        F_test * sizeof(float)
                    );
                    curr_token_idx++;

                    if (!block->active_k.empty()) {
                        int active_len = block->active_k.size() / F_test;
                        std::memcpy(
                            session->active_k_dense[l].data() + curr_token_idx * F_test,
                            block->active_k.data(),
                            block->active_k.size() * sizeof(float)
                        );
                        std::memcpy(
                            session->active_v_dense[l].data() + curr_token_idx * F_test,
                            block->active_v.data(),
                            block->active_v.size() * sizeof(float)
                        );
                        curr_token_idx += active_len;
                    }
                }
            }
            total_dense_tokens[l] = curr_token_idx;
        }

        {
            auto & b_list = runtime_manager_->get_ingest_manager().get_blocks(0);
            int curr_pos_idx = 0;
            std::fill(session->active_positions_dense.begin(), session->active_positions_dense.end(), 0);
            for (auto & block : b_list) {
                if (block->state == BlockState::DenseResident || block->state == BlockState::Compressing) {
                    for (int32_t t_pos : block->token_indices) {
                        session->active_positions_dense[curr_pos_idx++] = t_pos;
                    }
                }
            }
            total_positions = curr_pos_idx;
        }
    };
    
    // Format full prompt and tokenize
    std::vector<int32_t> prompt_tokens = model_->tokenize(req->prompt, true);
    if (prompt_tokens.empty()) {
        req->set_error("Empty tokenized prompt");
        return;
    }
    
    int L = prompt_tokens.size();
    if (L > n_slots * 64) {
        L = n_slots * 64;
        prompt_tokens.resize(L);
    }
    
    // Check if we can reuse the KV cache prefix from the previous turn
    int prefill_offset = 0;
    bool match = true;
    if (!session->last_turn_token_prefix.empty() && prompt_tokens.size() > session->last_turn_token_prefix.size()) {
        for (size_t i = 0; i < session->last_turn_token_prefix.size(); ++i) {
            if (prompt_tokens[i] != session->last_turn_token_prefix[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            prefill_offset = session->last_turn_token_prefix.size();
        }
    }
    
    if (!match || prefill_offset == 0) {
        // Clear global/session KVs since prefix did not match
        session->active_slot = 0;
        session->active_block_tokens = 0;
        for (auto& vec : session->active_k_dense) std::fill(vec.begin(), vec.end(), 0.0f);
        for (auto& vec : session->active_v_dense) std::fill(vec.begin(), vec.end(), 0.0f);
        session->persistent_k_dense.clear();
        session->persistent_v_dense.clear();
        session->inverted_index.clear();
        session->srl_state.ordered_slot_ids.clear();
        session->srl_state.sink_blocks.clear();
        session->srl_state.inverted_index.clear();
        session->srl_state.chunk_graph = diffkv::ChunkGraph();
        session->srl_state.semantic_index = diffkv::SemanticIndex();
        session->has_srl_state = false;
        session->layers_blocks.clear();
        session->pager_entries.clear();
        
        // Ensure residency clears the global manager active state
        session_manager_->ensure_residency(req->session_id);
        
        runtime_manager_->reset();
        
        // Determine adaptive micro-block size matching Python's logic
        int default_mbs = 64;
        if (const char* env_mbs = std::getenv("DIFFKV_MICRO_BLOCK_SIZE")) {
            default_mbs = std::stoi(env_mbs);
        }
        int adaptive_mbs = default_mbs;
        if (L > 0) {
            int raw_target = 16;
            if (L < 256) {
                raw_target = 16;
            } else if (L < 1024) {
                raw_target = 32;
            } else if (L < 4096) {
                raw_target = 64;
            } else if (L < 8192) {
                raw_target = 128;
            } else {
                raw_target = 256;
            }
            int target = std::min(raw_target, default_mbs);
            adaptive_mbs = std::max(16, ((target + 15) / 16) * 16);
        }
        session->micro_block_size = adaptive_mbs;
        runtime_manager_->set_micro_block_size(adaptive_mbs);
        
        prefill_offset = 0;
    }
    
    // ── 1. PREFILL PHASE ──
    int chunk_size = 2048;
    if (const char* env_pcs = std::getenv("DIFFKV_PREFILL_CHUNK_SIZE")) {
        try { chunk_size = std::stoi(env_pcs); } catch (...) {}
    }
    int pos_start = prefill_offset;
    int cached_len = prefill_offset;
    
    std::vector<std::vector<float>> k_activations(n_layers, std::vector<float>(L * F_test));
    std::vector<std::vector<float>> v_activations(n_layers, std::vector<float>(L * F_test));
    std::vector<float> prefill_output_logits(n_vocab);
    
    // Decompress prefix blocks from previous turns if cached_len > 0
    if (cached_len > 0) {
        for (int l = 0; l < n_layers; ++l) {
            auto & blocks = runtime_manager_->get_ingest_manager().get_blocks(l);
            for (auto & block : blocks) {
                if (block->anchor_idx >= cached_len) {
                    break;
                }
                int block_len = block->token_count();
                
                // Touch block to ensure residency
                if (block->state == BlockState::CPUResident) {
                    runtime_manager_->get_pager().touch(block.get(), runtime_manager_->get_engines());
                }
                
                int slot_id = block->pool_idx;
                if (block->state == BlockState::CompressedResident) {
                    auto & engine = runtime_manager_->get_engines()[l];
                    int rank = engine->get_U()->ne[0];
                    float scale_u = ggml_fp16_to_fp32(engine->get_host_U_scale()[slot_id]);
                    float block_scale = ggml_fp16_to_fp32(engine->get_host_scales()[slot_id]);
                    
                    for (int t = 0; t < block_len; ++t) {
                        int global_pos = block->anchor_idx + t;
                        if (global_pos >= cached_len) break;
                        
                        if (t == 0) {
                            for (int f = 0; f < F_test; ++f) {
                                k_activations[l][global_pos * F_test + f] = ggml_fp16_to_fp32(engine->get_host_anchors_K()[slot_id * F_test + f]);
                                v_activations[l][global_pos * F_test + f] = ggml_fp16_to_fp32(engine->get_host_anchors_V()[slot_id * F_test + f]);
                            }
                        } else {
                            int s = t - 1;
                            for (int f = 0; f < F_test; ++f) {
                                float sum_k = 0.0f;
                                float sum_v = 0.0f;
                                for (int r = 0; r < rank; ++r) {
                                    float u_val = (float)engine->get_host_U()[slot_id * 64 * rank + s * rank + r];
                                    float vk_val = ggml_fp16_to_fp32(engine->get_host_VK()[slot_id * rank * F_test + r * F_test + f]);
                                    float vv_val = ggml_fp16_to_fp32(engine->get_host_VV()[slot_id * rank * F_test + r * F_test + f]);
                                    sum_k += (u_val * scale_u) * vk_val;
                                    sum_v += (u_val * scale_u) * vv_val;
                                }
                                float anchor_k = ggml_fp16_to_fp32(engine->get_host_anchors_K()[slot_id * F_test + f]);
                                float anchor_v = ggml_fp16_to_fp32(engine->get_host_anchors_V()[slot_id * F_test + f]);
                                k_activations[l][global_pos * F_test + f] = anchor_k + sum_k * block_scale;
                                v_activations[l][global_pos * F_test + f] = anchor_v + sum_v * block_scale;
                            }
                        }
                    }
                } else {
                    // DenseResident or Compressing
                    for (int t = 0; t < block_len; ++t) {
                        int global_pos = block->anchor_idx + t;
                        if (global_pos >= cached_len) break;
                        
                        if (t == 0) {
                            std::memcpy(k_activations[l].data() + global_pos * F_test, block->anchor_k.data(), F_test * sizeof(float));
                            std::memcpy(v_activations[l].data() + global_pos * F_test, block->anchor_v.data(), F_test * sizeof(float));
                        } else {
                            int s = t - 1;
                            std::memcpy(k_activations[l].data() + global_pos * F_test, block->active_k.data() + s * F_test, F_test * sizeof(float));
                            std::memcpy(v_activations[l].data() + global_pos * F_test, block->active_v.data() + s * F_test, F_test * sizeof(float));
                        }
                    }
                }
            }
        }
    }
    
    while (pos_start < L) {
        int chunk_len = std::min(chunk_size, L - pos_start);
        int ctx_len   = pos_start + chunk_len;  // total KV context length
        
        size_t prior_bytes = (size_t)2 * n_layers * pos_start * F_test * sizeof(float);
        size_t graph_bytes = 8 * 1024 * 1024 + prior_bytes;
        
        struct ggml_init_params prefill_params = {
            /*.mem_size   =*/ graph_bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * prefill_ctx = ggml_init(prefill_params);
        if (!prefill_ctx) {
            req->set_error("Failed to initialize prefill context");
            return;
        }
        
        struct ggml_tensor * input_tokens_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
        ggml_set_input(input_tokens_prefill);
        struct ggml_tensor * positions_prefill = ggml_new_tensor_1d(prefill_ctx, GGML_TYPE_I32, chunk_len);
        ggml_set_input(positions_prefill);
        
        int intra_ctx_len = pos_start + chunk_len;
        struct ggml_tensor * mask_prefill = ggml_new_tensor_2d(prefill_ctx, GGML_TYPE_F16, intra_ctx_len, chunk_len);
        ggml_set_input(mask_prefill);
        
        std::vector<struct ggml_tensor *> prior_k_tensors(n_layers, nullptr);
        std::vector<struct ggml_tensor *> prior_v_tensors(n_layers, nullptr);
        bool has_prior = (pos_start > 0);
        if (has_prior) {
            int prior_intra_len = pos_start;
            for (int l = 0; l < n_layers; ++l) {
                prior_k_tensors[l] = ggml_new_tensor_3d(prefill_ctx, GGML_TYPE_F32,
                    head_dim, kv_heads, prior_intra_len);
                ggml_set_input(prior_k_tensors[l]);
                prior_v_tensors[l] = ggml_new_tensor_3d(prefill_ctx, GGML_TYPE_F32,
                    head_dim, kv_heads, prior_intra_len);
                ggml_set_input(prior_v_tensors[l]);
            }
        }
        
        struct ggml_tensor * prefill_logits = nullptr;
        std::vector<struct ggml_tensor *> prefill_k_layers(n_layers, nullptr);
        std::vector<struct ggml_tensor *> prefill_v_layers(n_layers, nullptr);
        bool is_last_chunk = (pos_start + chunk_len >= L);
        
        struct ggml_cgraph * prefill_graph = build_prefill_graph(
            prefill_ctx, *model_, input_tokens_prefill, positions_prefill, mask_prefill,
            has_prior ? &prior_k_tensors : nullptr,
            has_prior ? &prior_v_tensors : nullptr,
            &prefill_logits, &prefill_k_layers, &prefill_v_layers, is_last_chunk
        );
        if (is_last_chunk && prefill_logits) {
            ggml_set_output(prefill_logits);
        }
        
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, prefill_graph)) {
            ggml_free(prefill_ctx);
            req->set_error("Failed to allocate prefill graph memory");
            return;
        }
        
        ggml_backend_tensor_set(input_tokens_prefill, prompt_tokens.data() + pos_start, 0, chunk_len * sizeof(int32_t));
        
        std::vector<int32_t> pos_host(chunk_len);
        for (int i = 0; i < chunk_len; ++i) pos_host[i] = pos_start + i;
        ggml_backend_tensor_set(positions_prefill, pos_host.data(), 0, chunk_len * sizeof(int32_t));
        
        // Build full-context mask
        int intra_prior = pos_start;
        std::vector<ggml_fp16_t> mask_host(chunk_len * intra_ctx_len, ggml_fp32_to_fp16(0.0f));
        for (int qi = 0; qi < chunk_len; ++qi) {
            for (int kj = intra_prior; kj < intra_ctx_len; ++kj) {
                int chunk_kj = kj - intra_prior;
                if (chunk_kj > qi) {
                    mask_host[qi * intra_ctx_len + kj] = ggml_fp32_to_fp16(-INFINITY);
                }
            }
        }
        ggml_backend_tensor_set(mask_prefill, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));
        
        // Upload prior K/V context (from this turn's prior chunks)
        if (has_prior) {
            int intra_prior_len = pos_start;
            std::vector<int32_t> prior_positions(intra_prior_len);
            for (int t = 0; t < intra_prior_len; ++t) prior_positions[t] = t;
            
            int half_dim = head_dim / 2;
            std::vector<float> inv_freq(half_dim);
            for (int i = 0; i < half_dim; ++i) {
                inv_freq[i] = 1.0f / std::pow(model_->get_config().rope_freq_base, 2.0f * i / head_dim);
            }
            std::vector<float> cos_table(intra_prior_len * half_dim);
            std::vector<float> sin_table(intra_prior_len * half_dim);
            for (int t = 0; t < intra_prior_len; ++t) {
                float pos = (float)prior_positions[t];
                for (int i = 0; i < half_dim; ++i) {
                    float theta = pos * inv_freq[i];
                    cos_table[t * half_dim + i] = std::cos(theta);
                    sin_table[t * half_dim + i] = std::sin(theta);
                }
            }
            
            std::vector<float> prior_k_rotated(intra_prior_len * F_test);
            for (int l = 0; l < n_layers; ++l) {
                apply_rope_neox_cpu_fast(
                    k_activations[l].data(),
                    prior_k_rotated.data(),
                    cos_table.data(),
                    sin_table.data(),
                    intra_prior_len, kv_heads, head_dim
                );
                ggml_backend_tensor_set(prior_k_tensors[l],
                    prior_k_rotated.data(),
                    0, intra_prior_len * F_test * sizeof(float));
                ggml_backend_tensor_set(prior_v_tensors[l],
                    v_activations[l].data(),
                    0, intra_prior_len * F_test * sizeof(float));
            }
        }
        
        if (ggml_backend_sched_graph_compute(sched_, prefill_graph) != GGML_STATUS_SUCCESS) {
            ggml_free(prefill_ctx);
            req->set_error("Prefill compute execution failed");
            return;
        }
        
        std::vector<std::vector<float>> chunk_k(n_layers, std::vector<float>(chunk_len * F_test));
        std::vector<std::vector<float>> chunk_v(n_layers, std::vector<float>(chunk_len * F_test));
        for (int l = 0; l < n_layers; ++l) {
            ggml_backend_tensor_get(prefill_k_layers[l], chunk_k[l].data(), 0, chunk_len * F_test * sizeof(float));
            ggml_backend_tensor_get(prefill_v_layers[l], chunk_v[l].data(), 0, chunk_len * F_test * sizeof(float));
            std::memcpy(k_activations[l].data() + pos_start * F_test, chunk_k[l].data(), chunk_len * F_test * sizeof(float));
            std::memcpy(v_activations[l].data() + pos_start * F_test, chunk_v[l].data(), chunk_len * F_test * sizeof(float));
        }
        
        runtime_manager_->ingest_prefill(chunk_k, chunk_v, chunk_len, pos_start, prompt_tokens);
        
        if (pos_start + chunk_len >= L && prefill_logits) {
            ggml_backend_tensor_get(prefill_logits, prefill_output_logits.data(), 0, n_vocab * sizeof(float));
        }
        
        ggml_free(prefill_ctx);
        pos_start += chunk_len;
    }
    
    runtime_manager_->wait_for_compressor();
    runtime_manager_->update_descriptors(W_proj_host, desc_dim, head_dim);
    
    // Retrieve top prediction logit
    std::vector<std::pair<float, int>> prefill_top_k;
    for (int i = 0; i < n_vocab; ++i) {
        prefill_top_k.push_back({prefill_output_logits[i], i});
    }
    std::sort(prefill_top_k.begin(), prefill_top_k.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    int32_t first_decode_token = prefill_top_k[0].second;
    req->is_prefilled = true;
    
    // ── 2. DECODE PHASE Fresh Graph Setup ──
    struct ggml_init_params decode_params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * decode_ctx = ggml_init(decode_params);
    if (!decode_ctx) {
        req->set_error("Failed to initialize decode context");
        return;
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
    
#ifdef __APPLE__
    bool approx = true;
#else
    bool approx = false;
#endif
    if (const char* env_approx = std::getenv("DIFFKV_MPS_APPROXIMATE_ATTN")) {
        approx = (std::strcmp(env_approx, "1") == 0 || std::strcmp(env_approx, "true") == 0 || std::strcmp(env_approx, "yes") == 0 || std::strcmp(env_approx, "on") == 0);
    }
    
    std::vector<diffkv::CustomAttnUserData> userdata(n_layers);
    auto & kv_engines = runtime_manager_->get_engines();
    for (int l = 0; l < n_layers; ++l) {
        userdata[l].kv_engine = kv_engines[l].get();
        userdata[l].session_id = req->session_id;
        userdata[l].layer_idx = l;
        userdata[l].slot_indices = nullptr;
        userdata[l].n_q_heads = model_->get_config().n_head;
        userdata[l].n_kv_heads = model_->get_config().n_head_kv;
        userdata[l].rank = 32;
        userdata[l].S_max = 64;
        userdata[l].K = 0;
        userdata[l].D = head_dim;
        userdata[l].scale = 1.0f / std::sqrt((float)head_dim);
        userdata[l].has_rope = true;
        userdata[l].rope_freq_base = model_->get_config().rope_freq_base;
        userdata[l].approximate_attn = approx;
        userdata[l].ignore_c = false;
        userdata[l].srl_state = &session->srl_state;
        userdata[l].W_proj = W_proj_host.data();
        userdata[l].desc_dim = desc_dim;
    }
    
    struct ggml_tensor * decode_logits = nullptr;
    struct ggml_tensor * decode_selected_slots = nullptr;
    struct ggml_tensor * decode_concat_k = nullptr;
    struct ggml_tensor * decode_concat_v = nullptr;
    
    struct ggml_cgraph * decode_graph = build_decode_graph(
        decode_ctx, *model_, input_token_decode, position_decode, W_proj_decode,
        kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
        slots_mask_decode, host_slots_decode,
        srl_k_semantic, srl_k_keep,
        userdata.data(), &decode_logits, &decode_selected_slots,
        &decode_concat_k, &decode_concat_v
    );
    ggml_set_output(decode_logits);
    if (decode_selected_slots) ggml_set_output(decode_selected_slots);
    if (decode_concat_k) ggml_set_output(decode_concat_k);
    if (decode_concat_v) ggml_set_output(decode_concat_v);
    
    ggml_backend_sched_reset(sched_);
    if (decode_concat_k) ggml_backend_sched_set_tensor_backend(sched_, decode_concat_k, backend_);
    if (decode_concat_v) ggml_backend_sched_set_tensor_backend(sched_, decode_concat_v, backend_);
    if (!ggml_backend_sched_alloc_graph(sched_, decode_graph)) {
        ggml_free(decode_ctx);
        req->set_error("Failed to allocate memory for decode graph");
        return;
    }
    ggml_backend_tensor_set(W_proj_decode, W_proj_host.data(), 0, W_proj_host.size() * sizeof(float));
    
    int32_t last_token = first_decode_token;
    req->sfa_active = false;
    std::string first_piece = model_->token_to_piece(last_token);
    req->push_chunk(first_piece);
    req->generated_tokens.push_back(last_token);
    
    std::vector<int32_t> all_tokens = prompt_tokens;
    all_tokens.push_back(last_token);
    
    // Rebuild initial SRL state for the session
    session->srl_state.vsl_active_candidates.clear();
    session->srl_state.vsl_consecutive_helpers = 0;
    session->srl_state.ordered_slot_ids.clear();
    session->srl_state.sink_blocks.clear();
    session->srl_state.inverted_index.clear();
    session->srl_state.chunk_graph = diffkv::ChunkGraph();
    session->srl_state.semantic_index = diffkv::SemanticIndex();
    session->srl_state.recent_generated_tokens.clear();
    session->srl_state.current_query_tokens.clear();
    session->srl_state.current_step_slots.clear();
    session->srl_state.current_step_factual_tokens.clear();
    session->srl_state.current_step_count = 0;
    session->srl_state.recent_miss_rate = 0.0f;
    session->srl_state.k_multiplier = 1.0f;
    session->srl_state.call_count = 0;

    auto & blocks_layer0 = runtime_manager_->get_ingest_manager().get_blocks(0);
    std::vector<int32_t> compressed_slots;
    for (int i = 0; i < (int)blocks_layer0.size(); ++i) {
        if (blocks_layer0[i]->pool_idx != -1 &&
            (blocks_layer0[i]->state == BlockState::CompressedResident ||
             blocks_layer0[i]->state == BlockState::CPUResident)) {
            compressed_slots.push_back(blocks_layer0[i]->pool_idx); // Physical slot ID
        }
    }
    int completed_blocks = compressed_slots.size();
    if (completed_blocks > 0) {
        std::vector<float> desc_matrix_host(completed_blocks * desc_dim);
        for (int j = 0; j < completed_blocks; ++j) {
            int slot_id = compressed_slots[j];
            ggml_backend_tensor_get(
                runtime_manager_->get_engines()[0]->get_desc_matrix(),
                desc_matrix_host.data() + j * desc_dim,
                slot_id * desc_dim * sizeof(float),
                desc_dim * sizeof(float)
            );
        }

        session->srl_state = build_srl_state_from_blocks(
            desc_matrix_host.data(),
            compressed_slots.data(),
            completed_blocks,
            prompt_tokens.data(),
            L,
            session->micro_block_size + 1, // block_size
            stop_token_ids_,
            6, // K_semantic
            2, // K_temporal
            0.15f, // overlap_threshold
            true, // add_first_as_sink
            true  // add_last_as_sink
        );
        session->has_srl_state = true;

        std::unordered_set<int32_t> prime_slots(
            session->srl_state.chunk_graph.cluster_centers_tensor.begin(),
            session->srl_state.chunk_graph.cluster_centers_tensor.end()
        );
        session->srl_state.factual_store.build(
            k_activations,
            v_activations,
            prompt_tokens,
            W_proj_host.data(),
            desc_dim,
            head_dim,
            kv_heads,
            stop_token_ids_,
            session->srl_state.ordered_slot_ids,
            session->micro_block_size + 1,
            session->srl_state.inverted_index,
            prime_slots,
            true // use_salience_parser
        );
        session->srl_state.setup_sas_and_eqa(prompt_tokens, stop_token_ids_, [&](int32_t tid) {
            return model_->token_to_piece(tid);
        });
    }
    
    session->active_slot = runtime_manager_->get_ingest_manager().get_blocks(0).size() - 1;
    if (session->active_slot < 0) {
        session->active_slot = 0;
    }
    session->active_block_tokens = 0;
    if (!runtime_manager_->get_ingest_manager().get_blocks(0).empty()) {
        session->active_block_tokens = runtime_manager_->get_ingest_manager().get_blocks(0).back()->token_count();
        if (session->active_block_tokens > 0) {
            session->active_block_tokens--; // exclude anchor
        }
    }
    
    std::vector<int> dense_start_positions(n_layers, 0);
    std::vector<int> total_dense_tokens(n_layers, 0);
    sync_active_dense_buffers(dense_start_positions, total_dense_tokens);
    
    // Decode generation loop
    int retrieval_interval = 8;
    if (const char* env_ri = std::getenv("DIFFKV_RETRIEVAL_INTERVAL")) {
        retrieval_interval = std::max(1, std::stoi(env_ri));
    }
    std::vector<int32_t> cached_routed_blocks;
    std::vector<int32_t> cached_physical_candidates;
    int last_retrieval_step = -retrieval_interval; // force on first step
    int last_retrieval_active_slot = -1;

    for (int step = 1; step < req->max_tokens; ++step) {
        if (req->cancelled) {
            break;
        }
        if (session->active_slot >= n_slots) {
            break;
        }
        int current_pos = L + step - 1;
        
        bool do_retrieval = (step - last_retrieval_step >= retrieval_interval) || (session->active_slot != last_retrieval_active_slot);
        if (do_retrieval) {
            cached_routed_blocks = runtime_manager_->route_decode_slots(
                current_pos,
                all_tokens,
                session->srl_state,
                stop_token_ids_,
                srl_k_recency,
                srl_k_lexical,
                srl_k_graph,
                srl_k_host,
                session->active_slot
            );
            last_retrieval_step = step;
            last_retrieval_active_slot = session->active_slot;

            // Map block indices to physical pool slot indices
            cached_physical_candidates = cached_routed_blocks;
        }
        
        std::vector<int32_t> routed_blocks = cached_routed_blocks;
        std::vector<int32_t> physical_candidates = cached_physical_candidates;
        
        // Touch blocks to move from CPU/Disk to GPU VRAM
        runtime_manager_->touch_active_slots(routed_blocks);
        
        ggml_backend_tensor_set(input_token_decode, &last_token, 0, sizeof(last_token));
        ggml_backend_tensor_set(position_decode, &current_pos, 0, sizeof(current_pos));
        ggml_backend_tensor_set(host_slots_decode, physical_candidates.data(), 0, srl_k_host * sizeof(int32_t));
        
        std::vector<float> slots_mask_host(n_slots, -1e10f);
        int occupied_up_to = session->active_slot - 1;
        for (int i = 0; i <= occupied_up_to; ++i) {
            auto & blocks = runtime_manager_->get_ingest_manager().get_blocks(0);
            if (i >= 0 && i < (int)blocks.size()) {
                int slot_id = blocks[i]->pool_idx;
                if (slot_id >= 0 && slot_id < n_slots) {
                    if (blocks[i]->state != BlockState::DenseResident && blocks[i]->state != BlockState::Compressing) {
                        slots_mask_host[slot_id] = 0.0f;
                    }
                }
            }
        }
        ggml_backend_tensor_set(slots_mask_decode, slots_mask_host.data(), 0, n_slots * sizeof(float));
        
        int current_k = std::max(0, std::min(srl_k_keep, session->active_slot));
        int physical_active_slot = 0;
        auto & b_list = runtime_manager_->get_ingest_manager().get_blocks(0);
        if (session->active_slot >= 0 && session->active_slot < (int)b_list.size()) {
            physical_active_slot = b_list[session->active_slot]->pool_idx;
        }
        
        for (int l = 0; l < n_layers; ++l) {
            userdata[l].K = current_k;
            userdata[l].active_k_dense = session->active_k_dense[l].data();
            userdata[l].active_v_dense = session->active_v_dense[l].data();
            userdata[l].active_positions_dense = session->active_positions_dense.data();
            userdata[l].active_block_tokens = total_dense_tokens[l];
            userdata[l].active_slot = (total_dense_tokens[l] > 0) ? dense_start_positions[l] : current_pos;
            userdata[l].current_pos = current_pos;
        }
        
        if (ggml_backend_sched_graph_compute(sched_, decode_graph) != GGML_STATUS_SUCCESS) {
            req->set_error("Decode graph compute failed at step " + std::to_string(step));
            break;
        }
        
        std::vector<float> output_logits(n_vocab);
        ggml_backend_tensor_get(decode_logits, output_logits.data(), 0, n_vocab * sizeof(float));
        
        // Apply Factual Logit Bias
        for (int32_t tok_id : session->srl_state.current_step_factual_tokens) {
            if (tok_id >= 0 && tok_id < n_vocab) {
                output_logits[tok_id] += 3.0f;
            }
        }
        // Transition Biasing (Option 1)
        if (last_token >= 0 && !session->srl_state.current_step_factual_sequences.empty()) {
            std::unordered_set<int32_t> transition_candidates;
            for (const auto& seq : session->srl_state.current_step_factual_sequences) {
                if (seq.size() > 1) {
                    for (size_t i = 0; i < seq.size() - 1; ++i) {
                        if (seq[i] == last_token) {
                            transition_candidates.insert(seq[i + 1]);
                        }
                    }
                }
            }
            for (int32_t tok_id : transition_candidates) {
                if (tok_id >= 0 && tok_id < n_vocab) {
                    output_logits[tok_id] += 4.0f;
                }
            }
        }
        
        // Construct filtered penalty tokens matching Python runtime logic
        bool loop_detected = req->repetition_loop_detected;
        int penalty_window = loop_detected ? 256 : 64;
        float penalty_val = loop_detected ? std::max(req->repetition_penalty, 1.3f) : req->repetition_penalty;
        std::vector<int32_t> penalty_tokens;

        bool prompt_penalty_active = (!prompt_tokens.empty() && req->generated_tokens.size() < 8);
        if (!loop_detected && prompt_penalty_active) {
            penalty_val = std::max(penalty_val, 1.15f);
        }

        if (penalty_val != 1.0f) {
            thread_local std::vector<int8_t> alnum_cache;
            if (alnum_cache.empty()) {
                alnum_cache.assign(n_vocab, -1);
            }
            auto is_alphanumeric_token = [&](int32_t tok_id) -> bool {
                if (tok_id < 0 || tok_id >= n_vocab) return false;
                if (alnum_cache[tok_id] != -1) {
                    return alnum_cache[tok_id] == 1;
                }
                std::string piece = model_->token_to_piece(tok_id);
                bool has_alnum = false;
                for (char c : piece) {
                    if (std::isalnum(static_cast<unsigned char>(c))) {
                        has_alnum = true;
                        break;
                    }
                }
                alnum_cache[tok_id] = has_alnum ? 1 : 0;
                return has_alnum;
            };

            // 1. Add generated tokens (last penalty_window)
            int gen_start = std::max(0, (int)req->generated_tokens.size() - penalty_window);
            for (size_t i = gen_start; i < req->generated_tokens.size(); ++i) {
                int32_t tok = req->generated_tokens[i];
                if (is_alphanumeric_token(tok)) {
                    penalty_tokens.push_back(tok);
                }
            }

            // 2. Add prompt tokens (last 512) if prompt penalty is active
            if (!loop_detected && prompt_penalty_active) {
                int prompt_start = std::max(0, (int)prompt_tokens.size() - 512);
                for (size_t i = prompt_start; i < prompt_tokens.size(); ++i) {
                    int32_t tok = prompt_tokens[i];
                    if (is_alphanumeric_token(tok)) {
                        penalty_tokens.push_back(tok);
                    }
                }
            }
        }

        // SFA threshold raised to 0.55 to match main.cpp interactive path.
        if (session->srl_state.current_step_max_similarity >= 0.55f) {
            req->sfa_active = true;
        }

        // LM-VSL (Logit Masking) — graduated by retrieval confidence.
        // sim 0.55–0.69 → soft (-7): model can escape if LM distribution is strong.
        // sim ≥ 0.70    → hard (-1e10): verbatim extraction — the model must enter
        //   factual sequences from their first token (sequence-start-only fallback
        //   in get_allowed_tokens_vsl_cpp) and advance in order, fixing entity binding.
        if (req->sfa_active) {
            const auto& helper_ids = diffkv::get_helper_token_ids_cpp(*model_);
            auto allowed = diffkv::get_allowed_tokens_vsl_cpp(session->srl_state, helper_ids);
            int n_vocab = model_->get_config().n_vocab;
            float max_sim = session->srl_state.current_step_max_similarity;
            for (int i = 0; i < n_vocab; ++i) {
                if (allowed.count(i) == 0) {
                    if (max_sim >= 0.70f) {
                        output_logits[i] = -1e10f;   // hard: verbatim
                    } else {
                        output_logits[i] -= 7.0f;    // soft: guided
                    }
                }
            }
        }

        // Sampling — temperature threshold raised to match SFA/VSL bar.
        float effective_temperature = req->temperature;
        if (session->srl_state.current_step_max_similarity >= 0.55f) {
            effective_temperature = req->temperature * (1.0f - session->srl_state.current_step_max_similarity * 0.95f);
        }
        int32_t next_token = sample_token(
            output_logits,
            effective_temperature,
            req->top_p,
            penalty_val,
            penalty_tokens
        );
        
        if (model_->is_eog_token(next_token) || next_token == model_->token_eos()) {
            break;
        }

        // Strict Factual Alignment (SFA) State Update and Loop Check
        if (req->sfa_active) {
            const auto& helper_ids = diffkv::get_helper_token_ids_cpp(*model_);
            diffkv::update_vsl_state_cpp(next_token, session->srl_state, helper_ids);
            
            if (session->srl_state.vsl_consecutive_helpers >= 16) {
                std::string uncertainty_str = " [uncertain: details missing in source]";
                std::vector<int32_t> uncertainty_toks = model_->tokenize(uncertainty_str, false);
                for (int32_t t : uncertainty_toks) {
                    req->push_chunk(model_->token_to_piece(t));
                    req->generated_tokens.push_back(t);
                    all_tokens.push_back(t);
                }
                break;
            }
        }

        // Factual Early Stopping (Option 2 Extension)
        bool stop_generation = false;
        if (session->srl_state.current_step_max_similarity >= 0.5f) {
            for (const auto& seq : session->srl_state.current_step_factual_sequences) {
                if (seq.size() >= 5 && next_token == seq.back()) {
                    stop_generation = true;
                    break;
                }
            }
        }
        if (stop_generation) {
            break;
        }
        
        std::string piece = model_->token_to_piece(next_token);
        req->push_chunk(piece);
        
        std::vector<float> concat_k_host(n_layers * F_test);
        std::vector<float> concat_v_host(n_layers * F_test);
        ggml_backend_tensor_get(decode_concat_k, concat_k_host.data(), 0, n_layers * F_test * sizeof(float));
        ggml_backend_tensor_get(decode_concat_v, concat_v_host.data(), 0, n_layers * F_test * sizeof(float));

        std::vector<std::vector<float>> decode_k(n_layers, std::vector<float>(F_test));
        std::vector<std::vector<float>> decode_v(n_layers, std::vector<float>(F_test));
        for (int l = 0; l < n_layers; ++l) {
            std::memcpy(decode_k[l].data(), concat_k_host.data() + l * F_test, F_test * sizeof(float));
            std::memcpy(decode_v[l].data(), concat_v_host.data() + l * F_test, F_test * sizeof(float));
        }
        
        runtime_manager_->ingest_decode(decode_k, decode_v, current_pos, all_tokens);

        // Update SRL state and dynamic anchors
        session->srl_state.update_generated_tokens(next_token);
        session->srl_state.update_query_segment(next_token);

        {
            int kv_heads = model_->get_config().n_head_kv;
            std::vector<float> k_avg(head_dim, 0.0f);
            for (int h = 0; h < kv_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    k_avg[d] += decode_k[0][h * head_dim + d];
                }
            }
            for (int d = 0; d < head_dim; ++d) {
                k_avg[d] /= kv_heads;
            }
            session->srl_state.recent_decode_keys.push_back(k_avg);
            if (session->srl_state.recent_decode_keys.size() > 512) {
                session->srl_state.recent_decode_keys.erase(session->srl_state.recent_decode_keys.begin());
            }

            int32_t current_slot_id = 0;
            if (!session->srl_state.ordered_slot_ids.empty()) {
                current_slot_id = session->srl_state.ordered_slot_ids.back();
            }
            session->srl_state.generated_token_slots.push_back(current_slot_id);
            session->srl_state.update_dynamic_anchors(stop_token_ids_);
        }

        // Manual append newly ingested token to host-side dense resident buffers
        for (int l = 0; l < n_layers; ++l) {
            int offset = total_dense_tokens[l] * F_test;
            if (offset + F_test <= (int)session->active_k_dense[l].size()) {
                std::memcpy(session->active_k_dense[l].data() + offset, decode_k[l].data(), F_test * sizeof(float));
                std::memcpy(session->active_v_dense[l].data() + offset, decode_v[l].data(), F_test * sizeof(float));
                total_dense_tokens[l]++;
            }
        }
        if (total_positions < (int)session->active_positions_dense.size()) {
            session->active_positions_dense[total_positions++] = current_pos;
        }
        
        int old_active_slot = session->active_slot;
        session->active_slot = runtime_manager_->get_ingest_manager().get_blocks(0).size() - 1;
        if (session->active_slot < 0) {
            session->active_slot = 0;
        }
        session->active_block_tokens = 0;
        if (!runtime_manager_->get_ingest_manager().get_blocks(0).empty()) {
            session->active_block_tokens = runtime_manager_->get_ingest_manager().get_blocks(0).back()->token_count();
            if (session->active_block_tokens > 0) {
                session->active_block_tokens--; // exclude anchor
            }
        }
        
        if (session->active_block_tokens == 0 && session->active_slot > old_active_slot) {
            runtime_manager_->wait_for_compressor();
            int prev_slot = old_active_slot;
            auto & blocks = runtime_manager_->get_ingest_manager().get_blocks(0);
            if (prev_slot >= 0 && prev_slot < (int)blocks.size()) {
                auto & block = blocks[prev_slot];
                
                // Update descriptors first
                runtime_manager_->update_descriptors(W_proj_host, desc_dim, head_dim);

                std::vector<float> desc(desc_dim);
                ggml_backend_tensor_get(runtime_manager_->get_engines()[0]->get_desc_matrix(), desc.data(), block->pool_idx * desc_dim * sizeof(float), desc_dim * sizeof(float));

                update_srl_from_compressed_block(
                    session->srl_state,
                    desc.data(),
                    block->pool_idx, // Physical slot ID
                    session->token_ids.data() + block->anchor_idx, // Correct token IDs
                    block->token_count(),
                    block->anchor_idx,
                    stop_token_ids_
                );

                // Rebuild chunk graph only over compressed slots
                auto & all_blocks = runtime_manager_->get_ingest_manager().get_blocks(0);
                std::vector<int32_t> cur_slots;
                for (int i = 0; i < (int)all_blocks.size(); ++i) {
                    if (all_blocks[i]->pool_idx != -1 &&
                        (all_blocks[i]->state == BlockState::CompressedResident ||
                         all_blocks[i]->state == BlockState::CPUResident)) {
                        cur_slots.push_back(all_blocks[i]->pool_idx); // Physical slot ID
                    }
                }
                int cur_N = cur_slots.size();
                std::vector<float> cur_desc_matrix(cur_N * desc_dim);
                for (int j = 0; j < cur_N; ++j) {
                    int slot_id = cur_slots[j];
                    ggml_backend_tensor_get(
                        runtime_manager_->get_engines()[0]->get_desc_matrix(),
                        cur_desc_matrix.data() + j * desc_dim,
                        slot_id * desc_dim * sizeof(float),
                        desc_dim * sizeof(float)
                    );
                }

                session->srl_state.chunk_graph = build_chunk_graph(
                    cur_desc_matrix.data(),
                    cur_slots.data(),
                    cur_N,
                    6, 2,
                    &session->srl_state.inverted_index,
                    0.15f
                );
            }
            // Sync dense buffers only when block finishes
            sync_active_dense_buffers(dense_start_positions, total_dense_tokens);
        }
        
        last_token = next_token;
        req->generated_tokens.push_back(next_token);
        all_tokens.push_back(next_token);

        // ── Repetition-loop detection (Fix 2) ────────────────────────────────
        // Check for n-gram loops every 10 tokens after the minimum warm-up period.
        int n_new = (int)req->generated_tokens.size();
        if (!req->repetition_loop_detected && n_new >= 30 && n_new % 10 == 0) {
            int window_size = std::min(80, n_new);
            if (window_size >= 6) {
                std::vector<int32_t> window(req->generated_tokens.end() - window_size, req->generated_tokens.end());
                
                struct Ngram5 {
                    int32_t tokens[5];
                    bool operator==(const Ngram5& other) const {
                        return tokens[0] == other.tokens[0] &&
                               tokens[1] == other.tokens[1] &&
                               tokens[2] == other.tokens[2] &&
                               tokens[3] == other.tokens[3] &&
                               tokens[4] == other.tokens[4];
                    }
                };
                
                std::vector<std::pair<Ngram5, int>> counts;
                int total_ngrams = window_size - 5 + 1;
                for (int i = 0; i < total_ngrams; ++i) {
                    Ngram5 ng;
                    for (int j = 0; j < 5; ++j) {
                        ng.tokens[j] = window[i + j];
                    }
                    bool found = false;
                    for (auto& p : counts) {
                        if (p.first == ng) {
                            p.second++;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        counts.push_back({ng, 1});
                    }
                }
                
                int max_count = 0;
                for (const auto& p : counts) {
                    if (p.second > max_count) {
                        max_count = p.second;
                    }
                }
                
                if (total_ngrams > 0 && (float)max_count / total_ngrams >= 0.35f) {
                    req->repetition_loop_detected = true;
                    req->loop_detection_idx = n_new;
                    std::cout << "[DiffKV C++] WARNING: repetition loop detected for session "
                              << req->session_id << " at token " << n_new
                              << ". Escalating penalty window to 256 tokens and strength to 1.3x." << std::endl;
                }
            }
        }
        
        // If a loop persists for more than 40 tokens after detection, terminate early.
        if (req->repetition_loop_detected && n_new - req->loop_detection_idx >= 40) {
            std::cout << "[DiffKV C++] WARNING: repetition loop for session " << req->session_id
                      << " persisted for 40 tokens after detection — forcing EOS." << std::endl;
            break;
        }
    }
    
    // Update prefix cache for multi-turn reuse
    session->last_turn_token_prefix = all_tokens;
    session->token_ids = all_tokens;
    
    // Save state back to disk
    session_manager_->save_session(req->session_id);
    
    ggml_free(decode_ctx);
    req->is_finished = true;
    req->finish_stream();
}

} // namespace diffkv
