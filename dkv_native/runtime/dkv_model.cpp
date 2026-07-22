#include "runtime/dkv_model.hpp"
#include "ggml-alloc.h"
#include <cstdio>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace dkv {

static bool ends_with(const std::string & str, const std::string & suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

DKVModel::DKVModel() {}

DKVModel::~DKVModel() {
    if (model_buffer_) {
        ggml_backend_buffer_free(model_buffer_);
    }
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

bool DKVModel::load_from_file(const std::string & filename, ggml_backend_t backend) {
    std::cerr << "[DKV Model] Loading model from GGUF file: " << filename << " ..." << std::endl;

    // Load llama_model and vocab for tokenizer (vocab_only to save RAM)
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;
    model_ = llama_model_load_from_file(filename.c_str(), model_params);
    if (!model_) {
        std::cerr << "[DKV Model] Error: Failed to load llama_model from " << filename << std::endl;
        return false;
    }
    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
        std::cerr << "[DKV Model] Error: Failed to get vocabulary from model" << std::endl;
        return false;
    }

    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ggml_ctx_,
    };

    gguf_ctx_ = gguf_init_from_file(filename.c_str(), params);
    if (!gguf_ctx_) {
        std::cerr << "[DKV Model] Error: Failed to open GGUF file: " << filename << std::endl;
        return false;
    }

    if (!ggml_ctx_) {
        std::cerr << "[DKV Model] Error: GGUF loader did not initialize ggml_context!" << std::endl;
        return false;
    }

    // Allocate all tensors in ggml_ctx_ on the backend buffer
    model_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend);
    if (!model_buffer_) {
        std::cerr << "[DKV Model] Error: Failed to allocate model tensors on backend buffer!" << std::endl;
        return false;
    }
    std::cerr << "[DKV Model] Allocated weight buffer of size " 
              << ggml_backend_buffer_get_size(model_buffer_) / (1024 * 1024) 
              << " MB on backend: " << ggml_backend_name(backend) << std::endl;

    // Memory-map the GGUF file for zero-copy weight transfer
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[DKV Model] Error: Failed to open model file for reading weights!" << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        std::cerr << "[DKV Model] Error: Failed to get model file status!" << std::endl;
        close(fd);
        return false;
    }
    size_t file_size = sb.st_size;

    void* mmap_data = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_data == MAP_FAILED) {
        std::cerr << "[DKV Model] Error: Failed to mmap model file!" << std::endl;
        close(fd);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx_);
    int64_t n_tensors = gguf_get_n_tensors(gguf_ctx_);
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx_, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx_, name);
        if (!tensor) continue;
        
        size_t offset = gguf_get_tensor_offset(gguf_ctx_, i);
        size_t size = gguf_get_tensor_size(gguf_ctx_, i);
        
        if (data_offset + offset + size > file_size) {
            std::cerr << "[DKV Model] Error: Tensor offset out of bounds for " << name << std::endl;
            munmap(mmap_data, file_size);
            close(fd);
            return false;
        }

        const uint8_t* tensor_ptr = (const uint8_t*)mmap_data + data_offset + offset;
        ggml_backend_tensor_set(tensor, tensor_ptr, 0, size);
    }

    munmap(mmap_data, file_size);
    close(fd);

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
        std::cerr << "[DKV Model] output.weight is tied with token_embd.weight" << std::endl;
    }

    if (!token_embd_) {
        std::cerr << "[DKV Model] Error: Failed to find token_embd.weight tensor!" << std::endl;
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
            std::cerr << "[DKV Model] Error: Missing weights in layer " << l << "!" << std::endl;
            return false;
        }
    }

    std::cerr << "[DKV Model] Loaded successfully." << std::endl;
    return true;
}

void DKVModel::print_info() const {
    std::fprintf(stderr, "── Model Config Info ──\n");
    std::fprintf(stderr, "  n_vocab:        %d\n", config_.n_vocab);
    std::fprintf(stderr, "  n_embd:         %d\n", config_.n_embd);
    std::fprintf(stderr, "  n_layer:        %d\n", config_.n_layer);
    std::fprintf(stderr, "  n_head:         %d\n", config_.n_head);
    std::fprintf(stderr, "  n_head_kv:      %d\n", config_.n_head_kv);
    std::fprintf(stderr, "  n_ff:           %d\n", config_.n_ff);
    std::fprintf(stderr, "  n_rot:          %d (head_dim: %d)\n", config_.n_rot, config_.n_embd / config_.n_head);
    std::fprintf(stderr, "  rms_norm_eps:   %e\n", config_.rms_norm_eps);
    std::fprintf(stderr, "  rope_freq_base: %f\n", config_.rope_freq_base);
    std::fprintf(stderr, "  n_ctx:          %d\n", config_.n_ctx);
    std::fprintf(stderr, "───────────────────────\n");
}

std::vector<int32_t> DKVModel::tokenize(const std::string & text, bool add_special) const {
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

std::string DKVModel::detokenize(const std::vector<int32_t> & tokens) const {
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

std::string DKVModel::token_to_piece(int32_t token) const {
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

bool DKVModel::is_eog_token(int32_t token) const {
    if (!vocab_) return false;
    return llama_vocab_is_eog(vocab_, token);
}

int32_t DKVModel::token_eos() const {
    if (!vocab_) return -1;
    return llama_vocab_eos(vocab_);
}

int32_t DKVModel::token_bos() const {
    if (!vocab_) return -1;
    return llama_vocab_bos(vocab_);
}

} // namespace dkv
