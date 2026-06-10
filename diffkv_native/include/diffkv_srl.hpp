#pragma once

#include "ggml.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <cmath>

namespace diffkv {

struct InvertedIndexOccurrence {
    int32_t slot_id;
    int32_t abs_pos;
    int32_t rel_pos;
};

struct InvertedIndex {
    std::unordered_map<int32_t, std::vector<InvertedIndexOccurrence>> occurrences;
    std::unordered_map<int32_t, std::unordered_map<int32_t, std::vector<int32_t>>> chunk_vocabularies;
    std::unordered_map<int32_t, float> idf;

    void add_block_tokens(int32_t slot_id, const std::vector<int32_t>& tokens, int32_t start_pos, const std::unordered_set<int32_t>& stop_tokens) {
        for (size_t rel_pos = 0; rel_pos < tokens.size(); ++rel_pos) {
            int32_t tok = tokens[rel_pos];
            if (stop_tokens.count(tok)) {
                continue;
            }
            int32_t abs_pos = start_pos + rel_pos;
            occurrences[tok].push_back({slot_id, abs_pos, (int32_t)rel_pos});
            chunk_vocabularies[slot_id][tok].push_back((int32_t)rel_pos);
        }
    }

    void recompute_idf(int32_t total_blocks) {
        idf.clear();
        float n_blocks = std::max(1.0f, (float)total_blocks);
        for (const auto& pair : occurrences) {
            int32_t tok = pair.first;
            std::unordered_set<int32_t> slots_containing;
            for (const auto& occ : pair.second) {
                slots_containing.insert(occ.slot_id);
            }
            float n_containing = (float)slots_containing.size();
            idf[tok] = std::log(n_blocks / n_containing) + 1.0f;
        }
    }

    void clear() {
        occurrences.clear();
        chunk_vocabularies.clear();
        idf.clear();
    }
};


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
    float scale,
    int k_keep
);

} // namespace diffkv
