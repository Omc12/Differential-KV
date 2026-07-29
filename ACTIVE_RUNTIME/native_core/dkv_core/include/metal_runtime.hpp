#pragma once
#include <torch/extension.h>

namespace dkv {

// Returns true if custom Metal decode attention runtime is available and compiled.
bool is_metal_available();

// Launches the custom Metal compute shader for fused Project-Then-Attend decode attention.
torch::Tensor last_debug_buffer();

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
    int rank,
    // Residual and Fact Anchor Override buffers (Track D)
    const torch::Tensor& res_pos_K,
    const torch::Tensor& res_val_K,
    const torch::Tensor& res_pos_V,
    const torch::Tensor& res_val_V,
    const torch::Tensor& fact_pos,
    const torch::Tensor& fact_val_K,
    const torch::Tensor& fact_val_V,
    // Dense window buffers
    const torch::Tensor& dense_K,
    const torch::Tensor& dense_V,
    const torch::Tensor& cos_dense,
    const torch::Tensor& sin_dense,
    // Partial RoPE (Qwen3.5/GLM-style partial_rotary_factor<1.0): only the
    // first rotary_dim dims of D are ever rotated, the rest pass through
    // unrotated. -1 (default) means "full rotary" (rotary_dim == D), which
    // is exact prior behavior for every model/caller that predates this.
    int rotary_dim = -1,
    // Full-sequence RoPE tables [max_pos, rotary_dim] (raw, NOT padded to
    // head_dim) + absolute anchor position per routed slot [K]. When all three
    // are supplied, residual-K / fact-K are rotated at their TRUE token
    // position (anchor + within-block offset) instead of the block anchor's --
    // the Metal port of the CUDA/Triton DKV_RESIDUAL_EXACT_ROPE fix. Defaulted
    // empty so existing callers keep the prior anchor-position behavior.
    const c10::optional<torch::Tensor>& cos_full = c10::nullopt,
    const c10::optional<torch::Tensor>& sin_full = c10::nullopt,
    const c10::optional<torch::Tensor>& anchor_pos = c10::nullopt
);

} // namespace dkv
