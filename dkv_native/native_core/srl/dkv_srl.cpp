#include "native_core/srl/dkv_srl.hpp"
#include <iostream>

struct ggml_tensor* g_dbg_scores = nullptr;
struct ggml_tensor* g_dbg_q_mean = nullptr;
struct ggml_tensor* g_dbg_desc = nullptr;

namespace dkv {

struct ggml_tensor * compute_query_desc(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,          // [head_dim, n_head]
    struct ggml_tensor * W_proj      // [head_dim, desc_dim]
) {
    // 1. Mean pool over all query heads (mean along rows): [head_dim]
    // Since ggml_mean reduces ne[0] (head_dim), we transpose first, make contiguous for Metal, take mean, and transpose back.
    struct ggml_tensor * q_mean = ggml_transpose(ctx, ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, Q)))); // [head_dim, 1]
    ggml_set_name(q_mean, "srl_q_mean");
    g_dbg_q_mean = q_mean;

    // 2. Project to descriptor space: W_proj [head_dim, desc_dim] x q_mean [head_dim, 1] -> [desc_dim, 1]
    struct ggml_tensor * desc = ggml_mul_mat(ctx, W_proj, q_mean); // [desc_dim, 1]
    ggml_set_name(desc, "srl_desc");
    g_dbg_desc = desc;

    // 3. L2-normalize to unit sphere
    struct ggml_tensor * norm_desc = ggml_l2_norm(ctx, desc, 1e-8f);
    ggml_set_name(norm_desc, "srl_norm_desc");

    return norm_desc;
}

struct ggml_tensor * semantic_search_topk(
    struct ggml_context * ctx,
    struct ggml_tensor * q_desc,      // [desc_dim, 1]
    struct ggml_tensor * desc_matrix, // [desc_dim, n_slots]
    struct ggml_tensor * slots_mask,  // [n_slots, 1]
    int k
) {
    // Dot product: desc_matrix [desc_dim, n_slots] x q_desc [desc_dim, 1] -> [n_slots, 1]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, desc_matrix, q_desc); // [n_slots, 1]
    ggml_set_name(scores, "srl_scores");

    // Add mask to scores
    int n_allocated = desc_matrix->ne[1];
    struct ggml_tensor * slots_mask_view = ggml_view_1d(ctx, slots_mask, n_allocated, 0);
    ggml_set_name(slots_mask_view, "srl_slots_mask_view");
    scores = ggml_add(ctx, scores, slots_mask_view);

    // Cast scores to F32 to ensure compatibility with ggml_argsort_top_k
    struct ggml_tensor * scores_f32 = ggml_cast(ctx, scores, GGML_TYPE_F32);
    ggml_set_name(scores_f32, "srl_scores_f32");
    g_dbg_scores = scores_f32;

    // Get top-k indices (sorted descending) - clamp to scores_f32->ne[0] to prevent GGML_ASSERT failure
    int k_clamped = std::min(k, (int)scores_f32->ne[0]);
    struct ggml_tensor * top_idx = ggml_argsort_top_k(ctx, scores_f32, k_clamped); // [k_clamped, 1]
    ggml_set_name(top_idx, "srl_top_idx");

    return top_idx;
}

struct ggml_tensor * anchor_screen(
    struct ggml_context * ctx,
    struct ggml_tensor * Q,               // [head_dim, n_head]
    struct ggml_tensor * anchors_K,       // [head_dim, kv_heads, n_slots]
    struct ggml_tensor * candidate_slots, // [M] I32 candidate slot IDs
    struct ggml_tensor * slots_mask,      // [n_slots] validity mask
    float scale,
    int k_keep
) {
    // 1. Mean query over heads -> [head_dim, 1]
    struct ggml_tensor * q_mean = ggml_transpose(ctx, ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, Q))));
    ggml_set_name(q_mean, "srl_q_mean");

    // Cast anchors_K to F32 because ggml_mean only supports F32 on CPU backend
    struct ggml_tensor * anchors_K_f32 = ggml_cast(ctx, anchors_K, GGML_TYPE_F32);
    ggml_set_name(anchors_K_f32, "srl_anchors_K_f32");

    // 2. Mean anchors_K over kv_heads: permute to [kv_heads, head_dim, n_slots] so ggml_mean reduces ne[0] (kv_heads)
    struct ggml_tensor * anc_perm = ggml_permute(ctx, anchors_K_f32, 1, 0, 2, 3); // [kv_heads, head_dim, n_slots]
    ggml_set_name(anc_perm, "srl_anc_perm");
    struct ggml_tensor * anc_mean_raw = ggml_mean(ctx, ggml_cont(ctx, anc_perm)); // [1, head_dim, n_slots]
    ggml_set_name(anc_mean_raw, "srl_anc_mean_raw");
    struct ggml_tensor * anc_mean = ggml_reshape_2d(ctx, anc_mean_raw, Q->ne[0], anchors_K->ne[2]); // [head_dim, n_slots]
    ggml_set_name(anc_mean, "srl_anc_mean");

    // 3. Gather candidate slots: [head_dim, M]
    struct ggml_tensor * anc_gather = ggml_get_rows(ctx, anc_mean, candidate_slots);
    ggml_set_name(anc_gather, "srl_anc_gather");

    // 4. Dot product with mean query: [head_dim, M] x [head_dim, 1] -> [M, 1]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, anc_gather, q_mean);
    ggml_set_name(scores, "srl_scores");

    // 5. Gather and apply slots validity mask to scores
    struct ggml_tensor * mask_2d = ggml_reshape_2d(ctx, slots_mask, 1, slots_mask->ne[0]);
    struct ggml_tensor * candidate_mask = ggml_get_rows(ctx, mask_2d, candidate_slots); // [1, M]
    struct ggml_tensor * candidate_mask_2d = ggml_reshape_2d(ctx, candidate_mask, candidate_slots->ne[0], 1); // [M, 1]
    scores = ggml_add(ctx, scores, candidate_mask_2d);

    // 7. Scale attention scores
    struct ggml_tensor * scaled_scores = ggml_scale(ctx, scores, scale);
    ggml_set_name(scaled_scores, "srl_scaled_scores");

    // Cast scaled scores to F32 to prevent runtime type mismatch in argmax
    struct ggml_tensor * scaled_scores_f32 = ggml_cast(ctx, scaled_scores, GGML_TYPE_F32);
    ggml_set_name(scaled_scores_f32, "srl_scaled_scores_f32");

    // 8. argsort top-k kept slots (descending order) - clamp to scaled_scores_f32->ne[0] to prevent GGML_ASSERT failure
    int k_keep_clamped = std::min(k_keep, (int)scaled_scores_f32->ne[0]);
    struct ggml_tensor * top_idx = ggml_argsort_top_k(ctx, scaled_scores_f32, k_keep_clamped); // [k_clamped, 1]
    ggml_set_name(top_idx, "srl_top_idx");

    // Flatten top_idx to 1D
    struct ggml_tensor * top_idx_1d = ggml_reshape_1d(ctx, top_idx, top_idx->ne[0]);
    ggml_set_name(top_idx_1d, "srl_top_idx_1d");

    // 9. Return the actual candidate slot IDs corresponding to the top scores.
    // Since ggml_get_rows indexes along ne[1], we must reshape candidate_slots [M] to [1, M] first,
    // then cast to F32, perform get_rows (yielding [1, k_keep]), cast back to I32, and reshape to [k_keep].
    struct ggml_tensor * candidate_slots_2d  = ggml_reshape_2d(ctx, candidate_slots, 1, candidate_slots->ne[0]);
    ggml_set_name(candidate_slots_2d, "srl_candidate_slots_2d");
    struct ggml_tensor * candidate_slots_f32 = ggml_cast(ctx, candidate_slots_2d, GGML_TYPE_F32);
    ggml_set_name(candidate_slots_f32, "srl_candidate_slots_f32");
    struct ggml_tensor * filtered_slots_f32  = ggml_get_rows(ctx, candidate_slots_f32, top_idx_1d);
    ggml_set_name(filtered_slots_f32, "srl_filtered_slots_f32");
    struct ggml_tensor * filtered_slots_2d   = ggml_cast(ctx, filtered_slots_f32, GGML_TYPE_I32);
    ggml_set_name(filtered_slots_2d, "srl_filtered_slots_2d");
    struct ggml_tensor * filtered_slots      = ggml_reshape_1d(ctx, filtered_slots_2d, top_idx_1d->ne[0]);
    ggml_set_name(filtered_slots, "srl_filtered_slots");

    return filtered_slots;
}

} // namespace dkv
