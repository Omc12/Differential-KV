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

    // Vectorized flat representations
    std::vector<int32_t> parent_to_children_tensor;
    std::vector<int32_t> slot_to_parent_tensor;
    int max_children = 1;

    // Concentric zoning fields
    std::vector<int32_t> role_mapping_tensor;         // [max_slot + 1] int32 role (0=outer, 1=around, 2=center)
    std::vector<int32_t> cluster_centers_tensor;      // [C] int32 slot IDs of cluster centers
    std::vector<int32_t> slot_to_center_tensor;       // [max_slot + 1] int32 slot IDs of center

    ChunkGraph() : N(0), max_degree(1), max_children(1) {}

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
    const std::vector<int>*    block_anchor_idxs  = nullptr,  // anchor positions (hierarchical)
    int                        cached_len         = 0
) {
    ChunkGraph g;
    g.N = N;

    // Determine max slot ID early for concentric tensor sizing
    int32_t max_slot = 0;
    for (int i = 0; i < N; ++i) {
        if (slot_ids[i] > max_slot) {
            max_slot = slot_ids[i];
        }
    }

    g.role_mapping_tensor.assign(max_slot + 1, -1);
    g.slot_to_center_tensor.assign(max_slot + 1, -1);

    if (N == 0) {
        g.max_degree = 1;
        g.neighbors.assign(1, -1);
        g.weights.assign(1, 0.0f);
        return g;
    }

    // Initialize hierarchical structures
    g.parent_landmarks.resize(N, -1);
    g.slot_to_parent_tensor.assign(max_slot + 1, -1);

    if (block_pool_idxs && block_anchor_idxs
        && block_pool_idxs->size() == (size_t)N
        && block_anchor_idxs->size() == (size_t)N)
    {
        // Map: group_id -> first block row (becomes landmark representative)
        std::unordered_map<int, int32_t> group_to_landmark_row;
        int chunk_size = 512;
        int dynamic_chunk_size = static_cast<int>(std::sqrt(N * 128.0));
        if (dynamic_chunk_size < 256) dynamic_chunk_size = 256;
        if (dynamic_chunk_size > 1024) dynamic_chunk_size = 1024;
        chunk_size = dynamic_chunk_size;

        for (int i = 0; i < N; ++i) {
            int anchor = (*block_anchor_idxs)[i];
            int group_id = 0;
            if (cached_len > 0 && anchor >= cached_len) {
                int prefill_max_groups = (cached_len - 1) / chunk_size + 1;
                group_id = prefill_max_groups + ((anchor - cached_len) / chunk_size);
            } else {
                group_id = anchor / chunk_size;
            }
            if (group_to_landmark_row.find(group_id) == group_to_landmark_row.end()) {
                group_to_landmark_row[group_id] = i;
            }
        }

        for (int i = 0; i < N; ++i) {
            int anchor = (*block_anchor_idxs)[i];
            int group_id = 0;
            if (cached_len > 0 && anchor >= cached_len) {
                int prefill_max_groups = (cached_len - 1) / chunk_size + 1;
                group_id = prefill_max_groups + ((anchor - cached_len) / chunk_size);
            } else {
                group_id = anchor / chunk_size;
            }
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

        for (int i = 0; i < N; ++i) {
            int32_t child_slot = slot_ids[i];
            g.slot_to_parent_tensor[child_slot] = g.slot_to_parent[child_slot];
        }

        // Determine max children per group
        int max_children = 1;
        for (const auto& kv : g.parent_to_children) {
            int size = static_cast<int>(kv.second.size());
            if (size > max_children) {
                max_children = size;
            }
        }
        g.max_children = max_children;

        // Allocate parent_to_children_tensor
        g.parent_to_children_tensor.assign((max_slot + 1) * max_children, -1);
        for (const auto& kv : g.parent_to_children) {
            int32_t parent = kv.first;
            const auto& children = kv.second;
            if (parent >= 0 && parent <= max_slot) {
                int base_idx = parent * max_children;
                for (size_t c = 0; c < children.size(); ++c) {
                    g.parent_to_children_tensor[base_idx + c] = children[c];
                }
            }
        }
    }

    if (N == 1) {
        g.max_degree = 1;
        g.neighbors.assign(1, -1);
        g.weights.assign(1, 0.0f);
        
        int32_t slot_s = slot_ids[0];
        if (slot_s >= 0 && slot_s <= max_slot) {
            g.role_mapping_tensor[slot_s] = 2; // Center
            g.slot_to_center_tensor[slot_s] = slot_s;
        }
        g.cluster_centers_tensor = {slot_s};
        return g;
    }

    // ── Pairwise Cosine Similarity ──
    std::vector<float> sim(N * N, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, DESC_DIM,
                1.0f,
                desc_matrix, DESC_DIM,
                desc_matrix, DESC_DIM,
                0.0f,
                sim.data(), N);

    // ── Lexical setup ──
    std::vector<std::unordered_set<int>> vocabs(N);
    if (inv_index && !inv_index->chunk_vocabularies.empty()) {
        for (int idx = 0; idx < N; ++idx) {
            int32_t slot = slot_ids[idx];
            auto it = inv_index->chunk_vocabularies.find(slot);
            if (it != inv_index->chunk_vocabularies.end()) {
                for (const auto& kv : it->second) {
                    vocabs[idx].insert(kv.first);
                }
            }
        }
    }

    // ── Entity Isolation & Exclusion Pruning ──
    std::vector<int> sig_tokens(N, -1);
    std::vector<float> sig_idfs(N, -1.0f);
    if (inv_index && !inv_index->chunk_vocabularies.empty() && !inv_index->idf.empty()) {
        for (int i = 0; i < N; ++i) {
            int best_tok = -1;
            float best_idf = -1.0f;
            for (int tok : vocabs[i]) {
                float idf_val = 1.0f;
                auto idf_it = inv_index->idf.find(tok);
                if (idf_it != inv_index->idf.end()) {
                    idf_val = idf_it->second;
                }
                if (idf_val > best_idf) {
                    best_idf = idf_val;
                    best_tok = tok;
                }
            }
            sig_tokens[i] = best_tok;
            sig_idfs[i] = best_idf;
        }

        // Prune similarity matrix for mutually exclusive concept domains
        for (int i = 0; i < N; ++i) {
            int sig_i = sig_tokens[i];
            float idf_i = sig_idfs[i];
            if (sig_i == -1 || idf_i < 2.0f) continue;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                int sig_j = sig_tokens[j];
                float idf_j = sig_idfs[j];
                if (sig_j == -1 || idf_j < 2.0f) continue;
                if (sig_i != sig_j) {
                    bool has_cross_ref = vocabs[i].count(sig_j) || vocabs[j].count(sig_i);
                    if (!has_cross_ref) {
                        sim[i * N + j] = -1.0f;
                        sim[j * N + i] = -1.0f;
                    }
                }
            }
        }
    }

    // ── Handshake Hunting Protocol ──
    std::vector<std::vector<std::pair<int32_t, float>>> adj(N);
    for (int i = 0; i < N; ++i) {
        int anchor_i = (block_anchor_idxs && i < static_cast<int>(block_anchor_idxs->size())) ? (*block_anchor_idxs)[i] : 0;
        bool is_i_new = (cached_len > 0) && (anchor_i >= cached_len);
        
        // Chunks j < i target i (hunt)
        for (int j = 0; j < i; ++j) {
            int anchor_j = (block_anchor_idxs && j < static_cast<int>(block_anchor_idxs->size())) ? (*block_anchor_idxs)[j] : 0;
            bool is_j_new = (cached_len > 0) && (anchor_j >= cached_len);
            
            float sim_score = std::max(0.0f, sim[j * N + i]);
            
            float lex_score_i_to_j = 0.0f;
            float lex_score_j_to_i = 0.0f;
            
            bool is_excluded = false;
            if (inv_index && !inv_index->chunk_vocabularies.empty()) {
                int sig_i = sig_tokens[i];
                float idf_i = sig_idfs[i];
                int sig_j = sig_tokens[j];
                float idf_j = sig_idfs[j];
                if (sig_i != -1 && idf_i >= 2.0f && sig_j != -1 && idf_j >= 2.0f) {
                    if (sig_i != sig_j) {
                        bool has_cross_ref = vocabs[i].count(sig_j) || vocabs[j].count(sig_i);
                        if (!has_cross_ref) {
                            is_excluded = true;
                        }
                    }
                }
            }

            if (!is_excluded && inv_index && !inv_index->chunk_vocabularies.empty()) {
                const auto& v_i = vocabs[i];
                const auto& v_j = vocabs[j];
                if (!v_i.empty() || !v_j.empty()) {
                    float sum_idf_intersect = 0.0f;
                    float sum_idf_i = 0.0f;
                    float sum_idf_j = 0.0f;
                    
                    for (int tok : v_i) {
                        float idf_val = 1.0f;
                        auto idf_it = inv_index->idf.find(tok);
                        if (idf_it != inv_index->idf.end()) {
                            idf_val = idf_it->second;
                        }
                        sum_idf_i += idf_val;
                        if (v_j.count(tok)) {
                            sum_idf_intersect += idf_val;
                        }
                    }
                    for (int tok : v_j) {
                        float idf_val = 1.0f;
                        auto idf_it = inv_index->idf.find(tok);
                        if (idf_it != inv_index->idf.end()) {
                            idf_val = idf_it->second;
                        }
                        sum_idf_j += idf_val;
                    }
                    
                    if (sum_idf_intersect > 0.0f) {
                        if (sum_idf_i > 0.0f) {
                            lex_score_i_to_j = sum_idf_intersect / sum_idf_i;
                        }
                        if (sum_idf_j > 0.0f) {
                            lex_score_j_to_i = sum_idf_intersect / sum_idf_j;
                        }
                    }
                }
            }
            
            float temporal_boost = (std::abs(i - j) == 1) ? 0.2f : 0.0f;
            
            float weight_i_to_j = 0.5f * sim_score + 0.5f * lex_score_i_to_j + temporal_boost;
            float weight_j_to_i = 0.5f * sim_score + 0.5f * lex_score_j_to_i + temporal_boost;
            
            bool is_semantic_match = (sim_score >= 0.3f);
            bool is_lexical_match = (lex_score_i_to_j >= overlap_threshold) || (lex_score_j_to_i >= overlap_threshold);
            bool is_temporal_match = (std::abs(i - j) == 1);
            
            if (is_semantic_match || is_lexical_match || is_temporal_match) {
                bool add_i_to_j = true;
                bool add_j_to_i = true;
                
                if (cached_len > 0 && (is_i_new != is_j_new)) {
                    if (is_i_new) {
                        add_j_to_i = false; // older j cannot target newer i
                    } else {
                        add_i_to_j = false; // older i cannot target newer j
                    }
                }
                
                if (add_j_to_i) {
                    adj[j].push_back({i, std::max(1e-5f, weight_j_to_i)});
                }
                if (add_i_to_j) {
                    adj[i].push_back({j, std::max(1e-5f, weight_i_to_j)});
                }
            }
        }
    }

    // For backwards compatibility and robustness, add top semantic matches
    int k_sem = std::min(K_semantic, N - 1);
    for (int i = 0; i < N; ++i) {
        int anchor_i = (block_anchor_idxs && i < static_cast<int>(block_anchor_idxs->size())) ? (*block_anchor_idxs)[i] : 0;
        bool is_i_new = (cached_len > 0) && (anchor_i >= cached_len);

        std::vector<std::pair<float, int>> sims;
        sims.reserve(N - 1);
        const float* row = sim.data() + i * N;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            sims.push_back({row[j], j});
        }
        if (k_sem > 0 && !sims.empty()) {
            std::partial_sort(sims.begin(), sims.begin() + k_sem, sims.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int t = 0; t < k_sem; ++t) {
                int j = sims[t].second;
                int anchor_j = (block_anchor_idxs && j < static_cast<int>(block_anchor_idxs->size())) ? (*block_anchor_idxs)[j] : 0;
                bool is_j_new = (cached_len > 0) && (anchor_j >= cached_len);

                bool allow_i_to_j = (cached_len == 0) || (is_i_new == is_j_new) || (is_i_new && !is_j_new);
                bool allow_j_to_i = (cached_len == 0) || (is_i_new == is_j_new) || (is_j_new && !is_i_new);

                bool connected = false;
                for (const auto& p : adj[i]) {
                    if (p.first == j) {
                        connected = true;
                        break;
                    }
                }
                bool is_excluded = false;
                if (inv_index && !inv_index->chunk_vocabularies.empty()) {
                    int sig_i = sig_tokens[i];
                    float idf_i = sig_idfs[i];
                    int sig_j = sig_tokens[j];
                    float idf_j = sig_idfs[j];
                    if (sig_i != -1 && idf_i >= 2.0f && sig_j != -1 && idf_j >= 2.0f) {
                        if (sig_i != sig_j) {
                            bool has_cross_ref = vocabs[i].count(sig_j) || vocabs[j].count(sig_i);
                            if (!has_cross_ref) {
                                is_excluded = true;
                            }
                        }
                    }
                }

                if (allow_i_to_j && !connected && !is_excluded) {
                    float sim_score = std::max(0.0f, sim[i * N + j]);
                    float lex_score_i_to_j = 0.0f;
                    if (!is_excluded && inv_index && !inv_index->chunk_vocabularies.empty()) {
                        const auto& v_i = vocabs[i];
                        const auto& v_j = vocabs[j];
                        if (!v_i.empty() || !v_j.empty()) {
                            float sum_idf_intersect = 0.0f;
                            float sum_idf_i = 0.0f;
                            for (int tok : v_i) {
                                float idf_val = 1.0f;
                                auto idf_it = inv_index->idf.find(tok);
                                if (idf_it != inv_index->idf.end()) {
                                    idf_val = idf_it->second;
                                }
                                sum_idf_i += idf_val;
                                if (v_j.count(tok)) {
                                    sum_idf_intersect += idf_val;
                                }
                            }
                            if (sum_idf_intersect > 0.0f && sum_idf_i > 0.0f) {
                                lex_score_i_to_j = sum_idf_intersect / sum_idf_i;
                            }
                        }
                    }
                    float temporal_boost = (std::abs(i - j) == 1) ? 0.2f : 0.0f;
                    float weight_i_to_j = 0.5f * sim_score + 0.5f * lex_score_i_to_j + temporal_boost;
                    
                    adj[i].push_back({j, std::max(1e-5f, weight_i_to_j)});
                }
                
                bool reverse_connected = false;
                for (const auto& p : adj[j]) {
                    if (p.first == i) {
                        reverse_connected = true;
                        break;
                    }
                }
                if (allow_j_to_i && !reverse_connected && !is_excluded) {
                    float sim_score = std::max(0.0f, sim[i * N + j]);
                    float lex_score_j_to_i = 0.0f;
                    if (!is_excluded && inv_index && !inv_index->chunk_vocabularies.empty()) {
                        const auto& v_i = vocabs[i];
                        const auto& v_j = vocabs[j];
                        if (!v_i.empty() || !v_j.empty()) {
                            float sum_idf_intersect = 0.0f;
                            float sum_idf_j = 0.0f;
                            for (int tok : v_i) {
                                if (v_j.count(tok)) {
                                    float idf_val = 1.0f;
                                    auto idf_it = inv_index->idf.find(tok);
                                    if (idf_it != inv_index->idf.end()) {
                                        idf_val = idf_it->second;
                                    }
                                    sum_idf_intersect += idf_val;
                                }
                            }
                            for (int tok : v_j) {
                                float idf_val = 1.0f;
                                auto idf_it = inv_index->idf.find(tok);
                                if (idf_it != inv_index->idf.end()) {
                                    idf_val = idf_it->second;
                                }
                                sum_idf_j += idf_val;
                            }
                            if (sum_idf_intersect > 0.0f && sum_idf_j > 0.0f) {
                                lex_score_j_to_i = sum_idf_intersect / sum_idf_j;
                            }
                        }
                    }
                    float temporal_boost = (std::abs(i - j) == 1) ? 0.2f : 0.0f;
                    float weight_j_to_i = 0.5f * sim_score + 0.5f * lex_score_j_to_i + temporal_boost;
                    adj[j].push_back({i, std::max(1e-5f, weight_j_to_i)});
                }
            }
        }
    }

    // Deduplicate and determine actual max_degree, pad rows with -1
    int actual_max = 0;
    for (int i = 0; i < N; ++i) {
        auto& row_adj = adj[i];
        std::vector<std::pair<int32_t, float>> uniq;
        std::unordered_set<int32_t> seen;
        for (const auto& p : row_adj) {
            if (p.first != i && seen.find(p.first) == seen.end()) {
                uniq.push_back(p);
                seen.insert(p.first);
            }
        }
        row_adj = std::move(uniq);
        std::sort(row_adj.begin(), row_adj.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        actual_max = std::max(actual_max, static_cast<int>(row_adj.size()));
    }
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
    }

    // ── Concentric Cluster Relevance Zoning ──
    std::vector<int32_t> cluster_centers_list;
    if (block_pool_idxs && block_anchor_idxs
        && block_pool_idxs->size() == (size_t)N
        && block_anchor_idxs->size() == (size_t)N) {
        std::unordered_map<int, int32_t> group_to_landmark_row;
        for (int i = 0; i < N; ++i) {
            int group_id = (*block_anchor_idxs)[i] / 512;
            if (group_to_landmark_row.find(group_id) == group_to_landmark_row.end()) {
                group_to_landmark_row[group_id] = i;
            }
        }
        for (const auto& kv : group_to_landmark_row) {
            cluster_centers_list.push_back(slot_ids[kv.second]);
        }
    } else {
        // Fallback: window of 4
        for (int i = 0; i < N; i += 4) {
            cluster_centers_list.push_back(slot_ids[i]);
        }
    }

    g.cluster_centers_tensor = cluster_centers_list;
    std::unordered_set<int32_t> center_set(cluster_centers_list.begin(), cluster_centers_list.end());

    for (int32_t c : cluster_centers_list) {
        if (c >= 0 && c <= max_slot) {
            g.role_mapping_tensor[c] = 2; // Center
            g.slot_to_center_tensor[c] = c;
        }
    }

    for (int i = 0; i < N; ++i) {
        int32_t s = slot_ids[i];
        if (center_set.count(s)) continue;

        int32_t c = -1;
        if (block_pool_idxs && block_anchor_idxs && g.slot_to_parent.count(s)) {
            c = g.slot_to_parent[s];
        } else {
            int32_t best_center = -1;
            int best_dist = 999999;
            for (int32_t center_candidate : cluster_centers_list) {
                int dist = std::abs(s - center_candidate);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_center = center_candidate;
                }
            }
            c = best_center;
        }

        if (c != -1 && s >= 0 && s <= max_slot) {
            g.slot_to_center_tensor[s] = c;

            int row_s = i;
            int row_c = -1;
            for (int idx = 0; idx < N; ++idx) {
                if (slot_ids[idx] == c) {
                    row_c = idx;
                    break;
                }
            }

            bool is_around = false;
            if (row_c != -1) {
                float similarity_sc = sim[row_s * N + row_c];
                bool is_direct_neighbor = false;
                for (const auto& p : adj[row_c]) {
                    if (p.first == row_s) {
                        is_direct_neighbor = true;
                        break;
                    }
                }
                if (!is_direct_neighbor) {
                    for (const auto& p : adj[row_s]) {
                        if (p.first == row_c) {
                            is_direct_neighbor = true;
                            break;
                        }
                    }
                }

                if (similarity_sc >= 0.35f || is_direct_neighbor) {
                    is_around = true;
                }
            }

            g.role_mapping_tensor[s] = is_around ? 1 : 0;
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
    float                     hop_decay [[maybe_unused]],
    const std::vector<int32_t>* slot_ids = nullptr,
    const std::unordered_map<int32_t, float>* slot_activation_strength = nullptr
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

    // Apply retention scaling and slot reinforcement strength
    for (int i = 0; i < N; ++i) {
        float strength = 1.0f;
        if (slot_ids && slot_activation_strength && i < static_cast<int>(slot_ids->size())) {
            int32_t slot = (*slot_ids)[i];
            auto it = slot_activation_strength->find(slot);
            if (it != slot_activation_strength->end()) {
                strength = it->second;
            }
        }
        out[i] *= retention[i] * strength;
    }

    return out;
}

} // namespace diffkv
