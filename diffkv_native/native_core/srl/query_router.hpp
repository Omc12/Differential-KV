// query_router.hpp
// Translation of query_router.py to C++17.
// Provides the full SRL query routing pipeline:
//   adaptive_k -> semantic + lexical + graph + recency retrieval -> two-level gate.

#pragma once

#include "native_core/srl/session_srl_state.hpp"
#include "native_core/srl/chunk_descriptor.hpp"
#include "native_core/srl/chunk_graph.hpp"
#include "native_core/srl/inverted_index.hpp"
#include "native_core/srl/semantic_index.hpp"

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cassert>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace diffkv {

// ---------------------------------------------------------------------------
// BlockPoolInterface
//
// Minimal view into the block pool needed by the query router.
// Caller owns the memory.
// ---------------------------------------------------------------------------
struct BlockPoolInterface {
    float*   anchors_K;   // [max_blocks, kv_heads, head_dim] float32
                          //   (one representative anchor key per block)
    float*   W_proj;      // [DESC_DIM, head_dim] float32 projection matrix
    int      kv_heads;
    int      head_dim;
    int      max_blocks;
};

// ---------------------------------------------------------------------------
// two_level_gate
//
// Two-level anchor screening: cheaply rerank M candidates down to k_pass
// using per-block dot-product with the mean query vector.
// Applies an optional age-decay penalty:
//   penalty = srl_age_penalty * (age / max_age)  where age = position from end
//
// Q          : [H, D] float32 query
// candidates : [M] slot IDs to rerank
// Returns    : top k_pass slot IDs, sorted by descending anchor score
// ---------------------------------------------------------------------------
inline std::vector<int32_t> two_level_gate(
    const float*             Q,          // [H * D]
    int                      H,
    int                      D,
    const BlockPoolInterface& pool,
    const std::vector<int32_t>& candidates,
    float                    scale,
    int                      k_pass,
    const SessionSRLState*   srl_state   = nullptr
) {
    int M = static_cast<int>(candidates.size());
    if (M == 0 || k_pass <= 0) return {};
    int k_eff = std::min(k_pass, M);

    // Compute mean query vector: q_mean [D]
    std::vector<float> q_mean(D, 0.0f);
    float inv_H = 1.0f / static_cast<float>(H > 0 ? H : 1);
    for (int h = 0; h < H; ++h) {
        const float* qh = Q + h * D;
        for (int d = 0; d < D; ++d) q_mean[d] += qh[d] * inv_H;
    }

    // Build map: slot_id -> ordered_index for age penalty
    std::unordered_map<int32_t, int> slot_to_age;
    int max_age = 1;
    if (srl_state && !srl_state->ordered_slot_ids.empty()) {
        int n = static_cast<int>(srl_state->ordered_slot_ids.size());
        max_age = n;
        for (int i = 0; i < n; ++i) {
            // age = n - 1 - i  (0 = most recent)
            slot_to_age[srl_state->ordered_slot_ids[i]] = n - 1 - i;
        }
    }
    float age_penalty_coeff = srl_state ? srl_state->srl_age_penalty : 0.01f;

    // Score each candidate via dot product with its anchor key
    std::vector<std::pair<float, int32_t>> scored; // (score, slot_id)
    scored.reserve(M);

    for (int32_t slot : candidates) {
        // anchors_K layout: [block_idx, kv_heads, head_dim]
        // We need the row for this slot. Use slot as block_idx (caller convention).
        if (slot < 0 || slot >= pool.max_blocks) {
            scored.push_back({-1e9f, slot});
            continue;
        }

        const float* anchor = pool.anchors_K + (size_t)slot * pool.kv_heads * pool.head_dim;

        // Mean anchor over kv_heads -> [head_dim]
        std::vector<float> anchor_mean(pool.head_dim, 0.0f);
        float inv_kv = 1.0f / static_cast<float>(pool.kv_heads > 0 ? pool.kv_heads : 1);
        for (int kh = 0; kh < pool.kv_heads; ++kh) {
            const float* ah = anchor + kh * pool.head_dim;
            for (int d = 0; d < pool.head_dim; ++d)
                anchor_mean[d] += ah[d] * inv_kv;
        }

        // dot product with q_mean (both of dim head_dim == D)
        // If D != head_dim, use the smaller dimension
        int dim = std::min(D, pool.head_dim);
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d)
            dot += q_mean[d] * anchor_mean[d];
        dot *= scale;

        // Age penalty
        float penalty = 0.0f;
        if (srl_state) {
            auto it = slot_to_age.find(slot);
            if (it != slot_to_age.end()) {
                penalty = age_penalty_coeff
                        * static_cast<float>(it->second)
                        / static_cast<float>(max_age);
            }
        }

        scored.push_back({dot - penalty, slot});
    }

    // partial_sort to get top k_eff
    std::partial_sort(scored.begin(), scored.begin() + k_eff, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int32_t> result;
    result.reserve(k_eff);
    for (int i = 0; i < k_eff; ++i)
        result.push_back(scored[i].second);

    return result;
}

// ---------------------------------------------------------------------------
// adaptive_k
//
// Determine K (number of blocks to retrieve) based on query complexity.
//
// complexity = entropy(softmax(scores * 5)) / log(N)
// k_raw      = k_min + (k_max - k_min) * complexity
// k_scaled   = clamp(round(k_raw * k_multiplier), k_min, k_max)
// ---------------------------------------------------------------------------
inline int adaptive_k(
    const float*           q_desc,      // [DESC_DIM] float32
    const SessionSRLState& srl_state,
    int                    N_total
) {
    if (N_total <= 0) return srl_state.k_min;

    const SemanticIndex& sem = srl_state.semantic_index;
    int N = sem.N;
    if (N == 0) return srl_state.k_min;

    // Compute raw scores = dot(desc[i], q_desc) for all i
    std::vector<float> scores(N);
    sem.get_all_scores(q_desc, scores.data());

    // Softmax with temperature 5.0
    float temp = 5.0f;
    float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<float> probs(N);
    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        probs[i] = std::exp((scores[i] - max_score) * temp);
        sum_exp += probs[i];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int i = 0; i < N; ++i) probs[i] *= inv_sum;

    // Shannon entropy: H = -sum(p * log(p))
    float entropy = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (probs[i] > 1e-12f)
            entropy -= probs[i] * std::log(probs[i]);
    }

    // Normalise to [0,1] by log(N)
    float log_N      = std::log(static_cast<float>(N > 1 ? N : 2));
    float complexity = std::min(1.0f, entropy / log_N);

    float k_raw    = srl_state.k_min + (srl_state.k_max - srl_state.k_min) * complexity;
    float k_scaled = k_raw * srl_state.k_multiplier;

    int K = static_cast<int>(std::round(k_scaled));
    K = std::max(srl_state.k_min, std::min(srl_state.k_max, K));
    return K;
}

// ---------------------------------------------------------------------------
// route_query
//
// Full SRL query routing pipeline.  Selects K most relevant pool slots.
//
// Pipeline (mirrors Python query_router.py):
//   1.  Compute q_desc via compute_query_descriptor
//   2.  adaptive_k -> K
//   3.  Semantic: semantic_index.search(q_desc, k_semantic)
//   4.  Topic-switch detection: best_sem_score < 0.25
//   5.  Lexical: score all matching slots, keep top k_lexical; find rare_lex (IDF>=2)
//   6.  Graph expansion (2-hop) from semantic seed set
//   7.  Recency: last k_recency slots
//   8.  Merge (sink + semantic + rare_lex + graph + lexical + recency), dedup
//   9.  Two-level gate reranks non-sink candidates
//  10.  Return sink + filtered non-sink
// ---------------------------------------------------------------------------
inline std::vector<int32_t> route_query(
    const float*            Q,           // [H * D] float32 query (all query heads)
    int                     H,
    int                     D,
    SessionSRLState&        srl_state,
    const BlockPoolInterface& pool,
    float                   scale,
    int                     layer_idx [[maybe_unused]],
    const std::vector<int>* query_tokens = nullptr  // optional, for lexical routing
) {
    ++srl_state.call_count;
    int N = srl_state.semantic_index.N;

    // -------------------------------------------------------------------
    // Fallback: not enough blocks to use SRL
    // -------------------------------------------------------------------
    if (N < srl_state.routing_threshold) {
        return srl_state.ordered_slot_ids;
    }

    // -------------------------------------------------------------------
    // Step 1: Query descriptor
    // -------------------------------------------------------------------
    std::vector<float> q_desc(DESC_DIM, 0.0f);
    compute_query_descriptor(Q, pool.W_proj, H, D, q_desc.data());

    // -------------------------------------------------------------------
    // Step 2: Adaptive K
    // -------------------------------------------------------------------
    int K = adaptive_k(q_desc.data(), srl_state, N);

    // Decompose into per-channel budgets
    auto budget = srl_state.k_components(K);
    int k_semantic = std::max(1, budget.k_semantic);
    int k_lexical  = std::max(1, budget.k_lexical);
    int k_graph    = std::max(1, budget.k_graph);
    int k_recency  = std::max(1, budget.k_recency);

    // -------------------------------------------------------------------
    // Step 3: Semantic retrieval
    // -------------------------------------------------------------------
    auto sem_results = srl_state.semantic_index.search_with_scores(q_desc.data(), k_semantic);

    std::vector<int32_t> sem_slots;
    sem_slots.reserve(sem_results.size());
    float best_sem_score = sem_results.empty() ? 0.0f : sem_results[0].second;
    for (auto& [slot, score] : sem_results) sem_slots.push_back(slot);

    // -------------------------------------------------------------------
    // Step 4: Topic-switch detection
    // -------------------------------------------------------------------
    [[maybe_unused]] bool is_topic_switch = (best_sem_score < 0.25f);

    // -------------------------------------------------------------------
    // Step 5: Lexical retrieval + rare lexical
    // -------------------------------------------------------------------
    std::vector<int32_t> lex_slots;
    std::vector<int32_t> rare_lex_slots;

    // Use current_query_tokens if query_tokens not provided
    const std::vector<int>* tok_ptr = query_tokens;
    if (!tok_ptr && !srl_state.current_query_tokens.empty())
        tok_ptr = &srl_state.current_query_tokens;

    if (tok_ptr && !tok_ptr->empty()) {
        float decay = 0.999f;
        auto lex_scored = score_lexical_slots(srl_state.inverted_index, *tok_ptr, decay);

        int n_lex = static_cast<int>(lex_scored.size());
        int take_lex = std::min(k_lexical, n_lex);
        for (int i = 0; i < take_lex; ++i)
            lex_slots.push_back(lex_scored[i].first);

        // Rare lexical: slots whose best token has IDF >= 2.0
        std::unordered_set<int32_t> rare_seen;
        for (int tok : *tok_ptr) {
            auto idf_it = srl_state.inverted_index.idf.find(tok);
            if (idf_it == srl_state.inverted_index.idf.end()) continue;
            if (idf_it->second < 2.0f) continue;

            auto idx_it = srl_state.inverted_index.index.find(tok);
            if (idx_it == srl_state.inverted_index.index.end()) continue;
            for (int32_t s : idx_it->second) {
                if (!rare_seen.count(s)) {
                    rare_seen.insert(s);
                    rare_lex_slots.push_back(s);
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Step 6: Graph expansion (2-hop) from semantic seed set
    // -------------------------------------------------------------------
    std::vector<int32_t> graph_slots;
    const ChunkGraph& g = srl_state.chunk_graph;
    if (g.N > 0 && g.N == N) {
        // Build row-index lookup: slot_id -> row in graph/semantic_index
        // (assuming graph was built with the same ordering as semantic_index)
        const SemanticIndex& sem_idx = srl_state.semantic_index;

        // Get raw scores for all blocks (for retention / A0 initialisation)
        std::vector<float> sem_scores_cpu(N, 0.0f);
        sem_idx.get_all_scores(q_desc.data(), sem_scores_cpu.data());

        // Build set of seed row indices
        std::unordered_set<int32_t> seed_slots_set(sem_slots.begin(), sem_slots.end());
        std::vector<float> A0(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            if (seed_slots_set.count(sem_idx.slot_ids[i]))
                A0[i] = sem_scores_cpu[i];
        }

        // retention[i] = graph_hop_decay * sem_scores_cpu[i]
        std::vector<float> retention(N);
        for (int i = 0; i < N; ++i)
            retention[i] = srl_state.graph_hop_decay * sem_scores_cpu[i];

        // Hop 1: A1 = propagate(A0)
        std::vector<float> A1 = graph_propagate(g, A0, retention, srl_state.graph_hop_decay);
        // Hop 2: A2 = propagate(A1)
        std::vector<float> A2 = graph_propagate(g, A1, retention, srl_state.graph_hop_decay);

        // graph_scores = A1 + A2, exclude seed rows
        std::vector<std::pair<float, int32_t>> gscore_slots;
        for (int i = 0; i < N; ++i) {
            if (seed_slots_set.count(sem_idx.slot_ids[i])) continue;
            float gs = A1[i] + A2[i];
            if (gs > 0.0f)
                gscore_slots.push_back({gs, sem_idx.slot_ids[i]});
        }

        int k_graph_take = std::max(k_graph, 20);
        int n_graph = static_cast<int>(gscore_slots.size());
        int take_g  = std::min(k_graph_take, n_graph);
        if (take_g > 0) {
            std::partial_sort(gscore_slots.begin(), gscore_slots.begin() + take_g,
                              gscore_slots.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            for (int i = 0; i < take_g; ++i)
                graph_slots.push_back(gscore_slots[i].second);
        }
    }

    // -------------------------------------------------------------------
    // Step 7: Recency (last k_recency slots from ordered_slot_ids)
    // -------------------------------------------------------------------
    std::vector<int32_t> recency_slots;
    {
        const auto& ord = srl_state.ordered_slot_ids;
        int n_ord  = static_cast<int>(ord.size());
        int take_r = std::min(k_recency, n_ord);
        for (int i = n_ord - take_r; i < n_ord; ++i)
            recency_slots.push_back(ord[i]);
    }

    // -------------------------------------------------------------------
    // Step 8: Merge and deduplicate
    // Order: sink > semantic > rare_lex > graph > lexical > recency
    // -------------------------------------------------------------------
    std::unordered_set<int32_t> seen;
    std::vector<int32_t> sink_candidates;
    std::vector<int32_t> non_sink_candidates;

    // Helper: add unique slots to a destination vector
    auto add_unique = [&](const std::vector<int32_t>& src,
                          std::vector<int32_t>&       dst) {
        for (int32_t s : src) {
            if (!seen.count(s)) {
                seen.insert(s);
                dst.push_back(s);
            }
        }
    };

    // Sink blocks always included
    for (int32_t s : srl_state.sink_blocks) {
        if (!seen.count(s)) {
            seen.insert(s);
            sink_candidates.push_back(s);
        }
    }

    add_unique(sem_slots,      non_sink_candidates);
    add_unique(rare_lex_slots, non_sink_candidates);
    add_unique(graph_slots,    non_sink_candidates);
    add_unique(lex_slots,      non_sink_candidates);
    add_unique(recency_slots,  non_sink_candidates);

    // -------------------------------------------------------------------
    // Step 9: Two-level gate: rerank non-sink candidates
    // -------------------------------------------------------------------
    // We want to keep up to K - sink_count from non_sink
    int sink_count = static_cast<int>(sink_candidates.size());
    int non_sink_budget = std::max(1, K - sink_count);

    std::vector<int32_t> filtered_non_sink = two_level_gate(
        Q, H, D,
        pool,
        non_sink_candidates,
        scale,
        non_sink_budget,
        &srl_state
    );

    // -------------------------------------------------------------------
    // Step 10: Assemble final result
    // -------------------------------------------------------------------
    std::vector<int32_t> result;
    result.reserve(sink_candidates.size() + filtered_non_sink.size());
    result.insert(result.end(), sink_candidates.begin(),    sink_candidates.end());
    result.insert(result.end(), filtered_non_sink.begin(), filtered_non_sink.end());

    // Cache for this decode step
    srl_state.current_step_slots = result;
    srl_state.current_step_count = static_cast<int>(result.size());

    return result;
}

} // namespace diffkv
