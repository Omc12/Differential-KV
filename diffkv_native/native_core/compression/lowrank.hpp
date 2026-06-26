#pragma once

#include "native_core/diffkv_core/include/block_state.hpp"
#include "ggml.h"
#include <vector>
#include <unordered_set>

namespace diffkv {

// LowRank block compression parameters and outputs
struct LowRankCompressParams {
    int block_id;
    int block_size;  // S (total tokens in block, typically 64)
    int feat_dim;    // F
    int rank;        // R
    int pool_rank = 0; // The stride/capacity of the pool for this slot
    int pool_block_size = 64; // S_max of the pool slot
    int head_dim;    // D
    int anchor_idx;  // Sequence position of block start
    
    // Host-accessible inputs
    const float* raw_k_ptr;
    const float* raw_v_ptr;
    const int32_t* token_ids;
    const std::unordered_set<int32_t>* stop_token_ids;
    
    // Outputs in block pool
    int8_t* out_u_ptr;
    ggml_fp16_t* out_u_scale;
    ggml_fp16_t* out_u_row_scale = nullptr; // [S_max] per-token int8 scale (nullptr = legacy single-scale)
    ggml_fp16_t* out_vk_ptr;
    ggml_fp16_t* out_vv_ptr;
    ggml_fp16_t* out_scale;
    ggml_fp16_t* out_anchor_k;
    ggml_fp16_t* out_anchor_v;
    int32_t* out_seq_len;
    int32_t* out_anchor_position;
    int32_t* out_token_positions = nullptr;  // [S_max] true global seq position of each delta token (RoPE)

    // F9 sparse-residual outputs (optional; nullptr to skip). Sized [max_residual]
    // for positions and [max_residual * feat_dim] for values.
    int32_t* out_res_K_pos = nullptr;
    int32_t* out_res_V_pos = nullptr;
    ggml_fp16_t* out_res_K_val = nullptr;
    ggml_fp16_t* out_res_V_val = nullptr;
    int max_residual = 8;

    // Optional descriptor computation inside SVD compressor
    const float* W_proj = nullptr;
    int desc_dim = 0;
    float* out_desc = nullptr;
};

// Perform low-rank SVD compression on a block
bool compress_lowrank_block(const LowRankCompressParams& params);

// Internal SVD drivers (exposed for validation/testing if needed)
bool run_svd_driver(
    const float* A_input,
    int S, int F, int R,
    float* U_out,
    float* S_out,
    float* VT_out
);

} // namespace diffkv
