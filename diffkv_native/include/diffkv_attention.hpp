#pragma once

#include "ggml.h"
#include "diffkv_kv_engine.hpp"

namespace diffkv {

struct CustomAttnUserData {
    DiffKVKVEngine * kv_engine;
    struct ggml_tensor * slot_indices;
    int n_q_heads;
    int n_kv_heads;
    int rank;
    int S_max;
    int K;
    int D;
    float scale;
    bool has_rope;
    float rope_freq_base;
    const float * active_k_dense;
    const float * active_v_dense;
    int active_block_tokens;
    int active_slot;
};

// Custom attention launcher for Metal
void execute_metal_attention(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    struct ggml_tensor * slot_indices,
    CustomAttnUserData * data,
    float* lse_sparse_out
);

// GGML callback for custom attention operator
void custom_attention_op_callback(
    struct ggml_tensor * dst,
    const struct ggml_tensor * Q,
    const struct ggml_tensor * slot_indices,
    int ith,
    int nth,
    void * userdata
);

} // namespace diffkv
