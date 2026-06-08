#pragma once
#include <torch/extension.h>

namespace diffkv {

// Returns true if custom Metal decode attention runtime is available and compiled.
bool is_metal_available();

// Launches the custom Metal compute shader for fused Project-Then-Attend decode attention.
std::tuple<torch::Tensor, torch::Tensor> decode_attention_metal(
    const torch::Tensor& Q,               // [H_q, D]
    const torch::Tensor& U_pool,          // [N_pool, S_max, R]
    const torch::Tensor& U_scale_pool,    // [N_pool]
    const torch::Tensor& VK_pool,         // [N_pool, R, n_kv, D]
    const torch::Tensor& VV_pool,         // [N_pool, R, n_kv, D]
    const torch::Tensor& anchors_K,       // [N_pool, n_kv, D]
    const torch::Tensor& anchors_V,       // [N_pool, n_kv, D]
    const torch::Tensor& seq_lens,        // [N_pool]
    const torch::Tensor& scales,          // [N_pool]
    const torch::Tensor& cos_anc,         // [K]
    const torch::Tensor& sin_anc,         // [K]
    const torch::Tensor& slot_indices,    // [K]
    float scale,
    int n_q_heads,
    int n_kv_heads,
    int rank
);

} // namespace diffkv
