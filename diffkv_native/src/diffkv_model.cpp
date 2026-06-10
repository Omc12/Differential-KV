#include "diffkv_model.hpp"
#include <cstdio>
#include <iostream>

namespace diffkv {

static bool ends_with(const std::string & str, const std::string & suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

DiffKVModel::DiffKVModel() {}

DiffKVModel::~DiffKVModel() {
    if (model_) {
        llama_model_free(model_);
    }
    if (gguf_ctx_) {
        gguf_free(gguf_ctx_);
    }
    if (ggml_ctx_) {
        ggml_free(ggml_ctx_);
    }
}

bool DiffKVModel::load_from_file(const std::string & filename) {
    std::cout << "[DiffKV Model] Loading model from GGUF file: " << filename << " ..." << std::endl;

    // Load llama_model and vocab for tokenizer
    llama_model_params model_params = llama_model_default_params();
    model_ = llama_model_load_from_file(filename.c_str(), model_params);
    if (!model_) {
        std::cerr << "[DiffKV Model] Error: Failed to load llama_model from " << filename << std::endl;
        return false;
    }
    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
        std::cerr << "[DiffKV Model] Error: Failed to get vocabulary from model" << std::endl;
        return false;
    }

    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ggml_ctx_,
    };

    gguf_ctx_ = gguf_init_from_file(filename.c_str(), params);
    if (!gguf_ctx_) {
        std::cerr << "[DiffKV Model] Error: Failed to open GGUF file: " << filename << std::endl;
        return false;
    }

    if (!ggml_ctx_) {
        std::cerr << "[DiffKV Model] Error: GGUF loader did not initialize ggml_context!" << std::endl;
        return false;
    }

    // ── Parse configuration KVs (architecture independent) ────────────────────
    int64_t n_kv = gguf_get_n_kv(gguf_ctx_);
    for (int64_t i = 0; i < n_kv; ++i) {
        std::string key = gguf_get_key(gguf_ctx_, i);
        if (ends_with(key, ".block_count")) {
            config_.n_layer = gguf_get_val_u32(gguf_ctx_, i);
        } else if (ends_with(key, ".embedding_length")) {
            config_.n_embd = gguf_get_val_u32(gguf_ctx_, i);
        } else if (ends_with(key, ".feed_forward_length")) {
            config_.n_ff = gguf_get_val_u32(gguf_ctx_, i);
        } else if (ends_with(key, ".context_length")) {
            config_.n_ctx = gguf_get_val_u32(gguf_ctx_, i);
        } else if (ends_with(key, ".attention.head_count")) {
            config_.n_head = gguf_get_val_u32(gguf_ctx_, i);
        } else if (ends_with(key, ".attention.head_count_kv")) {
            config_.n_head_kv = gguf_get_val_u32(gguf_ctx_, i);
        } else if (ends_with(key, ".attention.layer_norm_rms_epsilon")) {
            config_.rms_norm_eps = gguf_get_val_f32(gguf_ctx_, i);
        } else if (ends_with(key, ".rope.freq_base")) {
            config_.rope_freq_base = gguf_get_val_f32(gguf_ctx_, i);
        } else if (ends_with(key, ".rope.dimension_count")) {
            config_.n_rot = gguf_get_val_u32(gguf_ctx_, i);
        }
    }

    // Fallbacks and defaults
    if (config_.n_head_kv == 0) {
        config_.n_head_kv = config_.n_head;
    }
    if (config_.n_rot == 0 && config_.n_head > 0) {
        config_.n_rot = config_.n_embd / config_.n_head; // default head_dim
    }

    // ── Get Global Tensors ───────────────────────────────────────────────────
    token_embd_ = ggml_get_tensor(ggml_ctx_, "token_embd.weight");
    output_norm_ = ggml_get_tensor(ggml_ctx_, "output_norm.weight");
    output_ = ggml_get_tensor(ggml_ctx_, "output.weight");
    if (!output_) {
        // Fallback name
        output_ = ggml_get_tensor(ggml_ctx_, "lm_head.weight");
    }
    if (!output_) {
        // Tied embeddings fallback: reuse token_embd.weight
        output_ = token_embd_;
        std::cout << "[DiffKV Model] output.weight is tied with token_embd.weight" << std::endl;
    }

    if (!token_embd_) {
        std::cerr << "[DiffKV Model] Error: Failed to find token_embd.weight tensor!" << std::endl;
        return false;
    }

    // Determine vocab size from embedding shape
    config_.n_vocab = token_embd_->ne[1];

    // ── Load Layer Tensors ────────────────────────────────────────────────────
    layers_.resize(config_.n_layer);
    for (int l = 0; l < config_.n_layer; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        
        layers_[l].wq = ggml_get_tensor(ggml_ctx_, (prefix + "attn_q.weight").c_str());
        layers_[l].bq = ggml_get_tensor(ggml_ctx_, (prefix + "attn_q.bias").c_str()); // optional
        
        layers_[l].wk = ggml_get_tensor(ggml_ctx_, (prefix + "attn_k.weight").c_str());
        layers_[l].bk = ggml_get_tensor(ggml_ctx_, (prefix + "attn_k.bias").c_str()); // optional
        
        layers_[l].wv = ggml_get_tensor(ggml_ctx_, (prefix + "attn_v.weight").c_str());
        layers_[l].bv = ggml_get_tensor(ggml_ctx_, (prefix + "attn_v.bias").c_str()); // optional
        
        layers_[l].wo = ggml_get_tensor(ggml_ctx_, (prefix + "attn_output.weight").c_str());
        layers_[l].bo = ggml_get_tensor(ggml_ctx_, (prefix + "attn_output.bias").c_str()); // optional

        layers_[l].attn_norm = ggml_get_tensor(ggml_ctx_, (prefix + "attn_norm.weight").c_str());
        layers_[l].ffn_norm  = ggml_get_tensor(ggml_ctx_, (prefix + "ffn_norm.weight").c_str());

        layers_[l].ffn_gate = ggml_get_tensor(ggml_ctx_, (prefix + "ffn_gate.weight").c_str());
        layers_[l].ffn_up   = ggml_get_tensor(ggml_ctx_, (prefix + "ffn_up.weight").c_str());
        layers_[l].ffn_down = ggml_get_tensor(ggml_ctx_, (prefix + "ffn_down.weight").c_str());

        // Verify critical layer tensors
        if (!layers_[l].wq || !layers_[l].wk || !layers_[l].wv || !layers_[l].wo ||
            !layers_[l].attn_norm || !layers_[l].ffn_norm ||
            !layers_[l].ffn_gate || !layers_[l].ffn_up || !layers_[l].ffn_down) {
            std::cerr << "[DiffKV Model] Error: Missing weights in layer " << l << "!" << std::endl;
            return false;
        }
    }

    std::cout << "[DiffKV Model] Loaded successfully." << std::endl;
    return true;
}

void DiffKVModel::print_info() const {
    std::printf("── Model Config Info ──\n");
    std::printf("  n_vocab:        %d\n", config_.n_vocab);
    std::printf("  n_embd:         %d\n", config_.n_embd);
    std::printf("  n_layer:        %d\n", config_.n_layer);
    std::printf("  n_head:         %d\n", config_.n_head);
    std::printf("  n_head_kv:      %d\n", config_.n_head_kv);
    std::printf("  n_ff:           %d\n", config_.n_ff);
    std::printf("  n_rot:          %d (head_dim: %d)\n", config_.n_rot, config_.n_embd / config_.n_head);
    std::printf("  rms_norm_eps:   %e\n", config_.rms_norm_eps);
    std::printf("  rope_freq_base: %f\n", config_.rope_freq_base);
    std::printf("  n_ctx:          %d\n", config_.n_ctx);
    std::printf("───────────────────────\n");
}

std::vector<int32_t> DiffKVModel::tokenize(const std::string & text, bool add_special) const {
    if (!vocab_) return {};
    std::vector<int32_t> tokens(text.length() + 4);
    int32_t n_tokens = llama_tokenize(vocab_, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab_, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, true);
    }
    if (n_tokens >= 0) {
        tokens.resize(n_tokens);
    } else {
        tokens.clear();
    }
    return tokens;
}

std::string DiffKVModel::detokenize(const std::vector<int32_t> & tokens) const {
    if (!vocab_ || tokens.empty()) return "";
    std::vector<char> text(tokens.size() * 16 + 128);
    int32_t n_chars = llama_detokenize(vocab_, tokens.data(), tokens.size(), text.data(), text.size(), false, false);
    if (n_chars < 0) {
        text.resize(-n_chars + 128);
        n_chars = llama_detokenize(vocab_, tokens.data(), tokens.size(), text.data(), text.size(), false, false);
    }
    if (n_chars >= 0) {
        return std::string(text.data(), n_chars);
    }
    return "";
}

std::string DiffKVModel::token_to_piece(int32_t token) const {
    if (!vocab_) return "";
    std::vector<char> buf(128);
    int32_t len = llama_token_to_piece(vocab_, token, buf.data(), buf.size(), 0, true);
    if (len < 0) {
        buf.resize(-len);
        len = llama_token_to_piece(vocab_, token, buf.data(), buf.size(), 0, true);
    }
    if (len >= 0) {
        return std::string(buf.data(), len);
    }
    return "";
}

bool DiffKVModel::is_eog_token(int32_t token) const {
    if (!vocab_) return false;
    return llama_vocab_is_eog(vocab_, token);
}

int32_t DiffKVModel::token_eos() const {
    if (!vocab_) return -1;
    return llama_vocab_eos(vocab_);
}

int32_t DiffKVModel::token_bos() const {
    if (!vocab_) return -1;
    return llama_vocab_bos(vocab_);
}

} // namespace diffkv
