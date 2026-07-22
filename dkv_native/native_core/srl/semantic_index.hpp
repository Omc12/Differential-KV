// semantic_index.hpp
// Translation of semantic_index.py to C++17.
// Maintains a matrix of L2-normalized block descriptors and supports
// fast cosine-similarity top-k search.

#pragma once

#include "native_core/srl/chunk_descriptor.hpp"  // DESC_DIM, l2_normalize

#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cassert>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace dkv {

// ---------------------------------------------------------------------------
// SemanticIndex
// ---------------------------------------------------------------------------
struct SemanticIndex {
    // desc_matrix[i * DESC_DIM ... (i+1)*DESC_DIM-1] : L2-normalised descriptor
    // for block at position i (corresponding to slot_ids[i]).
    std::vector<float>   desc_matrix;   // [N * DESC_DIM] float32
    std::vector<int32_t> slot_ids;      // [N] pool slot IDs
    int N = 0;                          // number of blocks

    // Sorted binary-search index for O(log N) slot_to_idx lookup
    std::vector<int32_t> sorted_slots;       // sorted copy of slot_ids
    std::vector<int32_t> sorted_to_orig;     // sorted_to_orig[i] = original row
    bool index_built = false;

    // ------------------------------------------------------------------
    // Build sorted index so slot_to_idx() can use binary search.
    // ------------------------------------------------------------------
    void build_sorted_index() {
        int n = static_cast<int>(slot_ids.size());
        sorted_slots.resize(n);
        sorted_to_orig.resize(n);

        // Create argsort of slot_ids
        std::iota(sorted_to_orig.begin(), sorted_to_orig.end(), 0);
        std::sort(sorted_to_orig.begin(), sorted_to_orig.end(),
                  [&](int32_t a, int32_t b) { return slot_ids[a] < slot_ids[b]; });

        for (int i = 0; i < n; ++i)
            sorted_slots[i] = slot_ids[sorted_to_orig[i]];

        index_built = true;
    }

    // ------------------------------------------------------------------
    // Returns the row index for a given slot_id, or -1 if not found.
    // Requires build_sorted_index() to have been called.
    // ------------------------------------------------------------------
    int slot_to_idx(int32_t slot_id) const {
        if (!index_built) return -1;
        auto it = std::lower_bound(sorted_slots.begin(), sorted_slots.end(), slot_id);
        if (it == sorted_slots.end() || *it != slot_id) return -1;
        int sorted_pos = static_cast<int>(it - sorted_slots.begin());
        return sorted_to_orig[sorted_pos];
    }

    // ------------------------------------------------------------------
    // search: find the top-k pool slot IDs by cosine similarity to q_desc.
    // Since descriptors are L2-normalised, cosine sim == dot product.
    //
    // Uses cblas_sgemv to compute all dot products in one call, then
    // nth_element to find the k largest scores.
    //
    // Returns up to k pool slot IDs (may return fewer if N < k).
    // ------------------------------------------------------------------
    std::vector<int32_t> search(const float* q_desc, int k) const {
        if (N == 0 || k <= 0) return {};
        int k_eff = std::min(k, N);

        // scores[i] = dot(desc_matrix[i*DESC_DIM..], q_desc)
        std::vector<float> scores(N, 0.0f);

        // desc_matrix: [N, DESC_DIM] row-major
        // y = A * x  where A=[N x DESC_DIM], x=q_desc[DESC_DIM]
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N, DESC_DIM,
                    1.0f, desc_matrix.data(), DESC_DIM,
                    q_desc, 1,
                    0.0f, scores.data(), 1);

        // Collect (score, original_row) pairs and partial-sort
        std::vector<std::pair<float, int>> scored(N);
        for (int i = 0; i < N; ++i) scored[i] = {scores[i], i};

        // nth_element is O(N) vs O(N log N) sort
        std::nth_element(scored.begin(), scored.begin() + k_eff, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        // Sort just the top-k by descending score for determinism
        std::sort(scored.begin(), scored.begin() + k_eff,
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<int32_t> result;
        result.reserve(k_eff);
        for (int i = 0; i < k_eff; ++i)
            result.push_back(slot_ids[scored[i].second]);

        return result;
    }

    // ------------------------------------------------------------------
    // search_with_scores: same as search but also returns the similarity
    // scores paired with slot IDs.
    // ------------------------------------------------------------------
    std::vector<std::pair<int32_t, float>> search_with_scores(const float* q_desc, int k) const {
        if (N == 0 || k <= 0) return {};
        int k_eff = std::min(k, N);

        std::vector<float> scores(N, 0.0f);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N, DESC_DIM,
                    1.0f, desc_matrix.data(), DESC_DIM,
                    q_desc, 1,
                    0.0f, scores.data(), 1);

        std::vector<std::pair<float, int>> scored(N);
        for (int i = 0; i < N; ++i) scored[i] = {scores[i], i};

        std::nth_element(scored.begin(), scored.begin() + k_eff, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::sort(scored.begin(), scored.begin() + k_eff,
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<std::pair<int32_t, float>> result;
        result.reserve(k_eff);
        for (int i = 0; i < k_eff; ++i)
            result.push_back({slot_ids[scored[i].second], scored[i].first});

        return result;
    }

    // ------------------------------------------------------------------
    // get_all_scores: returns raw dot-product scores for every block.
    // scores must be pre-allocated to [N].
    // ------------------------------------------------------------------
    void get_all_scores(const float* q_desc, float* scores_out) const {
        if (N == 0) return;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N, DESC_DIM,
                    1.0f, desc_matrix.data(), DESC_DIM,
                    q_desc, 1,
                    0.0f, scores_out, 1);
    }
};

// ---------------------------------------------------------------------------
// build_semantic_index
//
// Construct a SemanticIndex from a flat array of descriptors and slot IDs.
// Descriptors are copied; they should already be L2-normalised.
// ---------------------------------------------------------------------------
inline SemanticIndex build_semantic_index(
    const float*    desc_data,   // [N * DESC_DIM] float32, L2-normalised
    const int32_t*  slot_ids,    // [N] pool slot IDs
    int             N
) {
    SemanticIndex idx;
    idx.N = N;
    idx.desc_matrix.assign(desc_data, desc_data + (size_t)N * DESC_DIM);
    idx.slot_ids.assign(slot_ids, slot_ids + N);
    idx.build_sorted_index();
    return idx;
}

// ---------------------------------------------------------------------------
// add_block_to_index
//
// Append one new block to an existing SemanticIndex.
// desc: [DESC_DIM] float32, L2-normalised.
// ---------------------------------------------------------------------------
inline void add_block_to_index(
    SemanticIndex& idx,
    const float*   desc,
    int32_t        slot_id
) {
    idx.desc_matrix.insert(idx.desc_matrix.end(), desc, desc + DESC_DIM);
    idx.slot_ids.push_back(slot_id);
    ++idx.N;
    idx.build_sorted_index();  // rebuild for O(log N) lookup
}

} // namespace dkv
