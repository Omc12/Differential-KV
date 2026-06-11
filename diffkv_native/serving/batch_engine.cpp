#include "serving/batch_engine.hpp"
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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>

namespace diffkv {

// Helper to build the Qwen 2.5 dense prefill graph using causal flash attention
static struct ggml_cgraph * build_prefill_graph(
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

        // 5. Custom Metal Attention (wrapping forward/sparse Metal kernel)
        struct ggml_tensor * attn_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_embd);
        ggml_set_output(attn_out);

        // Bind custom attention operator callback
        struct ggml_tensor * attn_out_node = ggml_map_custom3(ctx, q_rope_flat, selected_slots, slots_mask, custom_attention_op_callback, GGML_OP_NONE, &userdata[l]);
        
        // 6. Output Projection (WO)
        struct ggml_tensor * attn_proj = ggml_mul_mat(ctx, layer.wo, attn_out_node);
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

    // Keep references to pre-rope layers
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
    
    int micro_block_size = 16;
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
    
    int srl_k_host = 1 + srl_k_recency + srl_k_lexical + 2 * srl_k_lexical + srl_k_graph;
    int F_test = kv_heads * head_dim;
    
    // Initialize W_proj_host
    std::vector<float> W_proj_host(head_dim * desc_dim);
    {
        std::mt19937 rand_gen(42);
        std::uniform_real_distribution<float> rand_dist(-1.0f, 1.0f);
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
        session->active_k_dense.assign(n_layers, std::vector<float>(64 * F_test, 0.0f));
        session->active_v_dense.assign(n_layers, std::vector<float>(64 * F_test, 0.0f));
    }
    if (session->seq_lens_by_layer.empty()) {
        session->seq_lens_by_layer.assign(n_layers, std::vector<int32_t>(n_slots, 0));
    }
    
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
        session->layers_blocks.clear();
        session->pager_entries.clear();
        
        // Ensure residency clears the global manager active state
        session_manager_->ensure_residency(req->session_id);
        
        runtime_manager_->reset();
        prefill_offset = 0;
    }
    
    // ── 1. PREFILL PHASE ──
    int chunk_size = 2048;
    int pos_start = prefill_offset;
    
    std::vector<std::vector<float>> k_activations(n_layers, std::vector<float>(L * F_test));
    std::vector<std::vector<float>> v_activations(n_layers, std::vector<float>(L * F_test));
    std::vector<float> prefill_output_logits(n_vocab);
    
    while (pos_start < L) {
        int chunk_len = std::min(chunk_size, L - pos_start);
        
        struct ggml_init_params prefill_params = {
            /*.mem_size   =*/ 4 * 1024 * 1024,
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
        struct ggml_tensor * mask_prefill = ggml_new_tensor_2d(prefill_ctx, GGML_TYPE_F16, chunk_len, chunk_len);
        ggml_set_input(mask_prefill);
        
        struct ggml_tensor * prefill_logits = nullptr;
        std::vector<struct ggml_tensor *> prefill_k_layers(n_layers, nullptr);
        std::vector<struct ggml_tensor *> prefill_v_layers(n_layers, nullptr);
        bool is_last_chunk = (pos_start + chunk_len >= L);
        
        struct ggml_cgraph * prefill_graph = build_prefill_graph(
            prefill_ctx, *model_, input_tokens_prefill, positions_prefill, mask_prefill,
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
        
        std::vector<ggml_fp16_t> mask_host(chunk_len * chunk_len, ggml_fp32_to_fp16(0.0f));
        for (int i = 0; i < chunk_len; ++i) {
            for (int j = i + 1; j < chunk_len; ++j) {
                mask_host[i * chunk_len + j] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        ggml_backend_tensor_set(mask_prefill, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));
        
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
    
    std::vector<diffkv::CustomAttnUserData> userdata(n_layers);
    auto & kv_engines = runtime_manager_->get_engines();
    for (int l = 0; l < n_layers; ++l) {
        userdata[l].kv_engine = kv_engines[l].get();
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
    }
    
    struct ggml_tensor * decode_logits = nullptr;
    struct ggml_tensor * decode_selected_slots = nullptr;
    std::vector<struct ggml_tensor *> decode_k_layers(n_layers, nullptr);
    std::vector<struct ggml_tensor *> decode_v_layers(n_layers, nullptr);
    
    struct ggml_cgraph * decode_graph = build_decode_graph(
        decode_ctx, *model_, input_token_decode, position_decode, W_proj_decode,
        kv_engines[0]->get_desc_matrix(), kv_engines[0]->get_anchors_K(),
        slots_mask_decode, host_slots_decode,
        srl_k_semantic, srl_k_keep,
        userdata.data(), &decode_logits, &decode_selected_slots,
        &decode_k_layers, &decode_v_layers
    );
    ggml_set_output(decode_logits);
    if (decode_selected_slots) ggml_set_output(decode_selected_slots);
    
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, decode_graph)) {
        ggml_free(decode_ctx);
        req->set_error("Failed to allocate memory for decode graph");
        return;
    }
    ggml_backend_tensor_set(W_proj_decode, W_proj_host.data(), 0, W_proj_host.size() * sizeof(float));
    
    int32_t last_token = first_decode_token;
    std::string first_piece = model_->token_to_piece(last_token);
    req->push_chunk(first_piece);
    req->generated_tokens.push_back(last_token);
    
    std::vector<int32_t> all_tokens = prompt_tokens;
    all_tokens.push_back(last_token);
    
    // Rebuild initial inverted index for the session
    session->inverted_index.clear();
    auto & blocks_layer0 = runtime_manager_->get_ingest_manager().get_blocks(0);
    int completed_blocks = blocks_layer0.size();
    for (int i = 0; i < completed_blocks; ++i) {
        if (i < completed_blocks - 1 || blocks_layer0[i]->token_count() == 1 + micro_block_size) {
            session->inverted_index.add_block_tokens(i, blocks_layer0[i]->token_indices, blocks_layer0[i]->anchor_idx, stop_token_ids_);
        }
    }
    session->inverted_index.recompute_idf(completed_blocks);
    
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
    
    // Copy active token dense embeddings to active buffers
    for (int l = 0; l < n_layers; ++l) {
        auto & b_list = runtime_manager_->get_ingest_manager().get_blocks(l);
        if (session->active_block_tokens > 0 && !b_list.empty()) {
            auto & last_block = b_list.back();
            std::fill(session->active_k_dense[l].begin(), session->active_k_dense[l].end(), 0.0f);
            std::fill(session->active_v_dense[l].begin(), session->active_v_dense[l].end(), 0.0f);
            std::memcpy(session->active_k_dense[l].data(), last_block->active_k.data(), last_block->active_k.size() * sizeof(float));
            std::memcpy(session->active_v_dense[l].data(), last_block->active_v.data(), last_block->active_v.size() * sizeof(float));
        } else {
            std::fill(session->active_k_dense[l].begin(), session->active_k_dense[l].end(), 0.0f);
            std::fill(session->active_v_dense[l].begin(), session->active_v_dense[l].end(), 0.0f);
        }
    }
    
    // Decode generation loop
    for (int step = 1; step < req->max_tokens; ++step) {
        if (req->cancelled) {
            break;
        }
        if (session->active_slot >= n_slots) {
            break;
        }
        int current_pos = L + step - 1;
        
        // Host Candidate Selection (SRL)
        std::vector<int32_t> routed_blocks = runtime_manager_->route_decode_slots(
            current_pos,
            all_tokens,
            session->inverted_index,
            stop_token_ids_,
            srl_k_recency,
            srl_k_lexical,
            srl_k_graph,
            srl_k_host,
            session->active_slot
        );
        
        // Touch blocks to move from CPU/Disk to GPU VRAM
        runtime_manager_->touch_active_slots(routed_blocks);
        
        // Map block indices to physical pool slot indices
        std::vector<int32_t> physical_candidates;
        for (int32_t block_idx : routed_blocks) {
            auto & blocks = runtime_manager_->get_ingest_manager().get_blocks(0);
            if (block_idx >= 0 && block_idx < (int)blocks.size()) {
                physical_candidates.push_back(blocks[block_idx]->pool_idx);
            } else {
                physical_candidates.push_back(0);
            }
        }
        
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
                    slots_mask_host[slot_id] = 0.0f;
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
            userdata[l].active_block_tokens = session->active_block_tokens;
            userdata[l].active_slot = physical_active_slot;
        }
        
        if (ggml_backend_sched_graph_compute(sched_, decode_graph) != GGML_STATUS_SUCCESS) {
            req->set_error("Decode graph compute failed at step " + std::to_string(step));
            break;
        }
        
        std::vector<float> output_logits(n_vocab);
        ggml_backend_tensor_get(decode_logits, output_logits.data(), 0, n_vocab * sizeof(float));
        
        // Sampling
        int32_t next_token = sample_token(
            output_logits,
            req->temperature,
            req->top_p,
            req->repetition_penalty,
            all_tokens
        );
        
        if (model_->is_eog_token(next_token) || next_token == model_->token_eos()) {
            break;
        }
        
        std::string piece = model_->token_to_piece(next_token);
        req->push_chunk(piece);
        
        std::vector<std::vector<float>> decode_k(n_layers, std::vector<float>(F_test));
        std::vector<std::vector<float>> decode_v(n_layers, std::vector<float>(F_test));
        for (int l = 0; l < n_layers; ++l) {
            ggml_backend_tensor_get(decode_k_layers[l], decode_k[l].data(), 0, F_test * sizeof(float));
            ggml_backend_tensor_get(decode_v_layers[l], decode_v[l].data(), 0, F_test * sizeof(float));
        }
        
        runtime_manager_->ingest_decode(decode_k, decode_v, current_pos, all_tokens);
        
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
        
        for (int l = 0; l < n_layers; ++l) {
            auto & b_list_l = runtime_manager_->get_ingest_manager().get_blocks(l);
            if (session->active_block_tokens > 0 && !b_list_l.empty()) {
                auto & last_block = b_list_l.back();
                std::memcpy(session->active_k_dense[l].data(), last_block->active_k.data(), last_block->active_k.size() * sizeof(float));
                std::memcpy(session->active_v_dense[l].data(), last_block->active_v.data(), last_block->active_v.size() * sizeof(float));
            } else {
                std::fill(session->active_k_dense[l].begin(), session->active_k_dense[l].end(), 0.0f);
                std::fill(session->active_v_dense[l].begin(), session->active_v_dense[l].end(), 0.0f);
            }
        }
        
        if (session->active_block_tokens == 0 && session->active_slot > old_active_slot) {
            int prev_slot = old_active_slot;
            auto & blocks = runtime_manager_->get_ingest_manager().get_blocks(0);
            if (prev_slot >= 0 && prev_slot < (int)blocks.size()) {
                auto & block = blocks[prev_slot];
                session->inverted_index.add_block_tokens(prev_slot, block->token_indices, block->anchor_idx, stop_token_ids_);
                session->inverted_index.recompute_idf(prev_slot + 1);
            }
            runtime_manager_->update_descriptors(W_proj_host, desc_dim, head_dim);
        }
        
        last_token = next_token;
        req->generated_tokens.push_back(next_token);
        all_tokens.push_back(next_token);
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
