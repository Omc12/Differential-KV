#pragma once

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include <string>
#include <vector>
#include <map>

namespace dkv {

struct ModelConfig {
    int n_vocab = 0;
    int n_embd = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_ff = 0;           // Feed-forward hidden dimension
    int n_rot = 0;          // RoPE rotation dimension (head_dim)
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    int n_ctx = 2048;
};

// Attention weights for a single layer
struct LayerWeights {
    struct ggml_tensor * wq = nullptr;
    struct ggml_tensor * bq = nullptr;  // Qwen 2.5 has QKV biases
    struct ggml_tensor * wk = nullptr;
    struct ggml_tensor * bk = nullptr;
    struct ggml_tensor * wv = nullptr;
    struct ggml_tensor * bv = nullptr;
    struct ggml_tensor * wo = nullptr;
    struct ggml_tensor * bo = nullptr;

    // RMSNorms
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * ffn_norm = nullptr;

    // FFN
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

class DKVModel {
public:
    DKVModel();
    ~DKVModel();

    bool load_from_file(const std::string & filename, ggml_backend_t backend);

    const ModelConfig & get_config() const { return config_; }
    
    // Weight access
    struct ggml_tensor * get_token_embd() { return token_embd_; }
    struct ggml_tensor * get_output_norm() { return output_norm_; }
    struct ggml_tensor * get_output() { return output_; }
    const std::vector<LayerWeights> & get_layers() const { return layers_; }

    // Tokenizer methods
    std::vector<int32_t> tokenize(const std::string & text, bool add_special) const;
    std::string detokenize(const std::vector<int32_t> & tokens) const;
    std::string token_to_piece(int32_t token) const;
    bool is_eog_token(int32_t token) const;
    int32_t token_eos() const;
    int32_t token_bos() const;

    // Prints model metadata and tensor info to stdout
    void print_info() const;

private:
    ModelConfig config_;
    struct ggml_context * ggml_ctx_ = nullptr;
    struct gguf_context * gguf_ctx_ = nullptr;
    struct llama_model * model_ = nullptr;
    const struct llama_vocab * vocab_ = nullptr;
    ggml_backend_buffer_t model_buffer_ = nullptr;

    // Global weights
    struct ggml_tensor * token_embd_ = nullptr;
    struct ggml_tensor * output_norm_ = nullptr;
    struct ggml_tensor * output_ = nullptr;

    // Layer weights
    std::vector<LayerWeights> layers_;
};

} // namespace dkv
