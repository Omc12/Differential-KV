// chunk_graph.hpp
// Translation of chunk_graph.py to C++17.
// Builds a mixed semantic/temporal/lexical similarity graph over KV blocks.
// Uses Accelerate cblas_sgemm for pairwise cosine similarity.

#pragma once

#include "native_core/srl/chunk_descriptor.hpp"   // DESC_DIM
#include "native_core/srl/inverted_index.hpp"     // InvertedTokenIndex, compute_block_overlap

#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <cstring>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

inline void cblas_sgemm(
    enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
    int M, int N, int K, float alpha, const float *A, int lda,
    const float *B, int ldb, float beta, float *C, int ldc
) {
    if (order != CblasRowMajor) return;
    if (transA == CblasNoTrans && transB == CblasTrans) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * lda + k] * B[j * ldb + k];
                }
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            }
        }
    }
}
#endif

namespace diffkv {

// ---------------------------------------------------------------------------
// ChunkGraph
// Flat CSR-like adjacency with fixed max_degree.
// neighbors[i * max_degree + j] == -1 means "no edge".
// ---------------------------------------------------------------------------
struct ChunkGraph {
    // Adjacency (row-major, size N x max_degree)
    std::vector<int32_t> neighbors;  // -1 = empty slot
    std::vector<float>   weights;    // edge weights

    int N          = 0;
    int max_degree = 1;

    // --------------- Hierarchical fields ---------------
    // parent_landmarks: pool slot IDs of each chunk's parent landmark block
    std::vector<int32_t> parent_landmarks;
    // parent_to_children: parent_slot -> [child_slot, ...]
    std::unordered_map<int32_t, std::vector<int32_t>> parent_to_children;
    // slot_to_parent: child_slot -> parent_slot
    std::unordered_map<int32_t, int32_t> slot_to_parent;

    ChunkGraph() : N(0), max_degree(1) {}

    // ---- Convenience accessors ----

    /// Number of valid (non -1) neighbors for row i.
    int degree(int i) const {
        int cnt = 0;
        int base = i * max_degree;
        for (int j = 0; j < max_degree; ++j)
            if (neighbors[base + j] >= 0) ++cnt;
        return cnt;
    }

    /// Get neighbor row index (or -1) at slot j of row i.
    int32_t neighbor(int i, int j) const {
        return neighbors[i * max_degree + j];
    }

    float weight(int i, int j) const {
        return weights[i * max_degree + j];
    }
};

// ---------------------------------------------------------------------------
// Internal: insert a neighbor into a sorted-by-weight adjacency list,
// keeping at most max_degree entries (highest weight first).
// adj[i]: vector<(row, weight)> mutable buffer
// ---------------------------------------------------------------------------
static inline void insert_neighbor(
    std::vector<std::pair<int32_t, float>>& adj,
    int32_t neighbor_row,
    float   w,
    int     max_degree
) {
    // Check if already present; update weight if higher
    for (auto& p : adj) {
        if (p.first == neighbor_row) {
            if (w > p.second) p.second = w;
            return;
        }
    }
    adj.push_back({neighbor_row, w});
    // Keep top max_degree by weight
    if (static_cast<int>(adj.size()) > max_degree) {
        std::sort(adj.begin(), adj.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        adj.resize(max_degree);
    }
}

// ---------------------------------------------------------------------------
// build_chunk_graph
//
// Constructs a ChunkGraph by:
//   1. Computing NxN pairwise cosine sim via cblas_sgemm (desc already normalised)
//   2. For each node: select top K_semantic semantic neighbours (excl. self)
//   3. Add temporal neighbours (i-1, i+1)
//   4. If inv_index has chunk_vocabularies, add lexical neighbours (Jaccard > overlap_threshold)
//   5. Merge/deduplicate, compute blended weights
//   6. Pad rows to max_degree with -1
//   7. Optionally build hierarchical parent groupings
// ---------------------------------------------------------------------------
inline ChunkGraph build_chunk_graph(
    const float*               desc_matrix,       // [N * DESC_DIM] float32, L2-normalised
    const int32_t*             slot_ids,          // [N] pool slot IDs
    int                        N,
    int                        K_semantic         = 6,
    int                        K_temporal         = 2,
    const InvertedTokenIndex*  inv_index          = nullptr,
    float                      overlap_threshold  = 0.15f,
    const std::vector<int32_t>* block_pool_idxs   = nullptr,  // for hierarchical grouping
    const std::vector<int>*    block_anchor_idxs  = nullptr   // anchor positions (hierarchical)
) {
    ChunkGraph g;
    g.N = N;

    if (N == 0) {
        g.max_degree = 1;
        g.neighbors.assign(1, -1);
        g.weights.assign(1, 0.0f);
        return g;
    }

    // ------------------------------------------------------------------
    // Step 1: Pairwise cosine similarity [N, N] via sgemm.
    // S = desc_matrix * desc_matrix^T  (both sides already L2-normalised)
    // S[i,j] = dot(desc[i], desc[j]) = cos_sim(i, j)
    // ------------------------------------------------------------------
    std::vector<float> sim(N * N, 0.0f);
    // cblas_sgemm: C = alpha * A * B + beta * C
    // A = desc_matrix [N, DESC_DIM], B = desc_matrix^T [DESC_DIM, N]
    // -> C [N, N]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, DESC_DIM,
                1.0f,
                desc_matrix, DESC_DIM,
                desc_matrix, DESC_DIM,
                0.0f,
                sim.data(), N);

    // ------------------------------------------------------------------
    // Step 2-5: Build per-node adjacency lists (exact blended formula)
    // ------------------------------------------------------------------
    std::vector<std::vector<std::pair<int32_t, float>>> adj(N);
    for (int i = 0; i < N; ++i) {
        std::unordered_set<int> unique_cands;

        // 1. Semantic top-K candidates
        std::vector<std::pair<float, int>> sims;
        sims.reserve(N - 1);
        const float* row = sim.data() + i * N;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            sims.push_back({row[j], j});
        }
        int k_eff = std::min(K_semantic, static_cast<int>(sims.size()));
        if (k_eff > 0) {
            std::partial_sort(sims.begin(), sims.begin() + k_eff, sims.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int t = 0; t < k_eff; ++t) {
                unique_cands.insert(sims[t].second);
            }
        }

        // 2. Temporal candidates
        for (int delta : {-1, +1}) {
            int j = i + delta;
            if (j >= 0 && j < N) {
                unique_cands.insert(j);
            }
        }

        // 3. Lexical candidates passing overlap_threshold
        if (inv_index && !inv_index->chunk_vocabularies.empty()) {
            int32_t slot_i = slot_ids[i];
            auto it_i = inv_index->chunk_vocabularies.find(slot_i);
            if (it_i != inv_index->chunk_vocabularies.end()) {
                const auto& vocab_i = it_i->second;
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    int32_t slot_j = slot_ids[j];
                    auto it_j = inv_index->chunk_vocabularies.find(slot_j);
                    if (it_j != inv_index->chunk_vocabularies.end()) {
                        float lex_score = compute_block_overlap(vocab_i, it_j->second);
                        if (lex_score >= overlap_threshold) {
                            unique_cands.insert(j);
                        }
                    }
                }
            }
        }

        // Calculate exact blended weights for all candidates
        for (int j : unique_cands) {
            float sem_score = std::max(0.0f, sim[i * N + j]);
            float lex_score = 0.0f;
            if (inv_index && !inv_index->chunk_vocabularies.empty()) {
                int32_t slot_i = slot_ids[i];
                int32_t slot_j = slot_ids[j];
                auto it_i = inv_index->chunk_vocabularies.find(slot_i);
                auto it_j = inv_index->chunk_vocabularies.find(slot_j);
                if (it_i != inv_index->chunk_vocabularies.end() && it_j != inv_index->chunk_vocabularies.end()) {
                    lex_score = compute_block_overlap(it_i->second, it_j->second);
                }
            }
            float temporal_boost = (std::abs(i - j) == 1) ? 0.2f : 0.0f;
            float w = 0.5f * sem_score + 0.5f * lex_score + temporal_boost;
            adj[i].push_back({j, w});
        }
    }

    // ------------------------------------------------------------------
    // Step 6: Determine actual max_degree, pad rows with -1
    // ------------------------------------------------------------------
    int actual_max = 0;
    for (int i = 0; i < N; ++i) {
        // Sort each adjacency list descending by weight and cap
        auto& row_adj = adj[i];
        std::sort(row_adj.begin(), row_adj.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        actual_max = std::max(actual_max, static_cast<int>(row_adj.size()));
    }
    // Ensure at least 1 to avoid zero-size allocation
    g.max_degree = std::max(1, actual_max);

    g.neighbors.assign((size_t)N * g.max_degree, -1);
    g.weights.assign((size_t)N * g.max_degree, 0.0f);

    for (int i = 0; i < N; ++i) {
        const auto& row_adj = adj[i];
        int base = i * g.max_degree;
        for (int j = 0; j < static_cast<int>(row_adj.size()); ++j) {
            g.neighbors[base + j] = row_adj[j].first;
            g.weights  [base + j] = row_adj[j].second;
        }
        // rest remain -1 / 0.0 from initialization
    }

    // ------------------------------------------------------------------
    // Step 7: Hierarchical grouping (optional)
    // Group blocks by anchor_idx // 512 as the "parent" landmark.
    // block_pool_idxs[i] = pool index for row i
    // block_anchor_idxs[i] = absolute anchor token position for row i
    // ------------------------------------------------------------------
    if (block_pool_idxs && block_anchor_idxs
        && block_pool_idxs->size() == (size_t)N
        && block_anchor_idxs->size() == (size_t)N)
    {
        g.parent_landmarks.resize(N, -1);

        // Map: group_id -> first block row (becomes landmark representative)
        std::unordered_map<int, int32_t> group_to_landmark_row;

        for (int i = 0; i < N; ++i) {
            int group_id = (*block_anchor_idxs)[i] / 512;
            if (group_to_landmark_row.find(group_id) == group_to_landmark_row.end()) {
                group_to_landmark_row[group_id] = i;
            }
        }

        for (int i = 0; i < N; ++i) {
            int group_id = (*block_anchor_idxs)[i] / 512;
            int landmark_row = group_to_landmark_row[group_id];
            int32_t parent_slot = slot_ids[landmark_row];
            int32_t child_slot  = slot_ids[i];

            g.parent_landmarks[i] = parent_slot;
            g.slot_to_parent[child_slot]         = parent_slot;
            g.parent_to_children[parent_slot].push_back(child_slot);
        }

        // Deduplicate children lists
        for (auto& kv : g.parent_to_children) {
            auto& vec = kv.second;
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
        }
    }

    return g;
}

// ---------------------------------------------------------------------------
// graph_propagate
//
// Two-hop graph signal propagation used in query routing.
//
// Given seed_scores[i] (nonzero at seed rows), propagates one hop:
//   out[j] += seed_scores[i] * weight(i, j) / damping[i]
//
// Returns out vector of length N.
// damping[i] = 1.0 + log(1.0 + degree(i))
// ---------------------------------------------------------------------------
inline std::vector<float> graph_propagate(
    const ChunkGraph&      g,
    const std::vector<float>& seed_scores,   // [N]
    const std::vector<float>& retention,     // [N] pointwise decay
    float                     hop_decay [[maybe_unused]]
) {
    int N = g.N;
    std::vector<float> out(N, 0.0f);
    if (N == 0) return out;

    for (int i = 0; i < N; ++i) {
        if (seed_scores[i] == 0.0f) continue;
        int deg = g.degree(i);
        float damping = 1.0f + std::log(1.0f + static_cast<float>(deg));
        int base = i * g.max_degree;
        for (int k = 0; k < g.max_degree; ++k) {
            int32_t j = g.neighbors[base + k];
            if (j < 0) break;
            out[j] += seed_scores[i] * g.weights[base + k] / damping;
        }
    }

    // Apply retention scaling
    for (int i = 0; i < N; ++i)
        out[i] *= retention[i];

    return out;
}

} // namespace diffkv
