#pragma once

#include "ggml.h"

namespace dkv {

// 1. compute_query_desc:
//    Pools all query heads: Q [H, D] -> q_mean [D].
//    Projects it: W_proj [desc_dim, D] x q_mean [D] -> desc [desc_dim].
//    L2-normalizes: returns [desc_dim] float32 descriptor.
struct ggml_tensor * compute_query_desc(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,          // [D, H]
    struct ggml_tensor * W_proj      // [D, desc_dim]
);

// 2. semantic_search_topk:
//    Dot product over descriptor matrix: desc_matrix [desc_dim, N] x q_desc [desc_dim] -> scores [N].
//    Returns top-k indices.
struct ggml_tensor * semantic_search_topk(
    struct ggml_context * ctx,
    struct ggml_tensor * q_desc,
    struct ggml_tensor * desc_matrix,
    struct ggml_tensor * slots_mask,
    int k
);

// 3. anchor_screen:
//    L1 anchor screening: rerank candidate slots using anchor key dot product.
//    Takes average query Q [D] and computes dot product with anchors_K [D, kv_heads, M] -> scores [M].
//    Returns top slot IDs.
struct ggml_tensor * anchor_screen(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,
    struct ggml_tensor * anchors_K,
    struct ggml_tensor * candidate_slots,
    struct ggml_tensor * slots_mask,
    float scale,
    int k_keep
);

} // namespace dkv
