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
    int head_dim;    // D
    int rank_min = 4;    // R_min
    int rank_max = 32;    // R_max
    
    // Host-accessible inputs
    const float* raw_k_ptr;
    const float* raw_v_ptr;
    const int32_t* token_ids;
    const std::unordered_set<int32_t>* stop_token_ids;
    
    // Outputs in block pool
    int8_t* out_u_ptr;
    ggml_fp16_t* out_u_scale;
    ggml_fp16_t* out_vk_ptr;
    ggml_fp16_t* out_vv_ptr;
    ggml_fp16_t* out_scale;
    ggml_fp16_t* out_anchor_k;
    ggml_fp16_t* out_anchor_v;
    int32_t* out_seq_len;
    bool* out_skip_compression = nullptr;
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
