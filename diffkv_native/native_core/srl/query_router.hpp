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
                penalty = age_penalty_coeff * static_cast<float>(it->second);
            }
        }

        // Reinforcement boost
        float boost = 0.0f;
        if (srl_state) {
            auto it = srl_state->slot_activation_strength.find(slot);
            if (it != srl_state->slot_activation_strength.end()) {
                boost = (it->second - 1.0f) * 0.1f;
            }
        }

        scored.push_back({dot - penalty + boost, slot});
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

    // Normalise to [0,1] by log(N_total) — matches Python: max_ent = log(N_total)
    float log_N_total = std::log(static_cast<float>(N_total > 1 ? N_total : 2));
    float complexity  = std::min(1.0f, entropy / std::max(log_N_total, 1e-8f));

    // ── k_min floor scales with context length (matches query_router.py:151) ──
    // k_max is capped at N_total so we never request more blocks than exist
    int k_max   = std::min(srl_state.k_max, N_total);
    int k_min   = std::min(std::max(srl_state.k_min, static_cast<int>(0.15f * N_total)), k_max);

    // Early return: pool is already small enough — attend to everything
    if (N_total <= k_min) return N_total;

    // Scale K linearly with complexity (use int() truncation like Python)
    int k_raw    = k_min + static_cast<int>((k_max - k_min) * complexity);
    int k_scaled = static_cast<int>(k_raw * srl_state.k_multiplier);

    // ── C_active cluster boost (matches query_router.py:176-194) ──
    // C_active = number of parent landmark blocks whose descriptor similarity
    // to q_desc is >= max(0.30, 0.85 * S_max).
    int C_active = 1;
    {
        const ChunkGraph& cg = srl_state.chunk_graph;
        if (!cg.parent_landmarks.empty()) {
            float S_max = 0.0f;
            std::unordered_set<int32_t> seen_parents;
            std::vector<float> parent_scores_vec;
            for (int32_t pslot : cg.parent_landmarks) {
                if (pslot < 0 || seen_parents.count(pslot)) continue;
                seen_parents.insert(pslot);
                int row = srl_state.semantic_index.slot_to_idx(pslot);
                if (row < 0) continue;
                const float* desc = srl_state.semantic_index.desc_matrix.data() + (size_t)row * DESC_DIM;
                float dot = 0.0f;
                for (int d = 0; d < DESC_DIM; ++d) dot += desc[d] * q_desc[d];
                parent_scores_vec.push_back(dot);
                S_max = std::max(S_max, dot);
            }
            if (!parent_scores_vec.empty()) {
                float theta = std::max(0.30f, 0.85f * S_max);
                int n_active = 0;
                for (float s : parent_scores_vec) {
                    if (s >= theta) ++n_active;
                }
                C_active = std::max(1, n_active);
            }
        }
    }
    // Apply the cluster boost: k_scaled *= (1 + 0.35 * ln(C_active))
    k_scaled = static_cast<int>(k_scaled * (1.0f + 0.35f * std::log(static_cast<float>(C_active))));

    return std::max(k_min, std::min(k_max, k_scaled));
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
    int K = adaptive_k(q_desc.data(), srl_state, srl_state.n_active_blocks());

    // Decompose into per-channel budgets
    auto budget = srl_state.k_components(K);
    int k_semantic = std::max(1, budget.k_semantic);
    int k_lexical  = std::max(1, budget.k_lexical);
    int k_graph    = std::max(1, budget.k_graph);
    int k_recency  = std::max(1, budget.k_recency);

    // -------------------------------------------------------------------
    // Step 3: Semantic retrieval (Hierarchical Graph-based Routing)
    // -------------------------------------------------------------------
    std::vector<int32_t> sem_slots;
    float best_sem_score = 0.00f;
    
    bool concentric_routed = false;
    const ChunkGraph& cg = srl_state.chunk_graph;
    
    if (!cg.cluster_centers_tensor.empty() && !cg.role_mapping_tensor.empty()) {
        // 1. Score all active cluster centers (centroids)
        std::vector<int32_t> centers;
        std::vector<int> center_rows;
        for (int32_t slot : cg.cluster_centers_tensor) {
            int row = srl_state.semantic_index.slot_to_idx(slot);
            if (row >= 0) {
                centers.push_back(slot);
                center_rows.push_back(row);
            }
        }
        
        if (!center_rows.empty()) {
            std::vector<std::pair<float, int32_t>> center_scores;
            center_scores.reserve(center_rows.size());
            
            for (size_t idx = 0; idx < center_rows.size(); ++idx) {
                int row = center_rows[idx];
                int32_t slot = centers[idx];
                
                float dot = 0.0f;
                const float* desc = srl_state.semantic_index.desc_matrix.data() + (size_t)row * DESC_DIM;
                for (int d = 0; d < DESC_DIM; ++d) {
                    dot += desc[d] * q_desc[d];
                }
                center_scores.push_back({dot, slot});
            }
            
            // Sort center scores descending
            std::sort(center_scores.begin(), center_scores.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            
            best_sem_score = center_scores[0].first;
            
            // Select top cluster centers
            int k_centers = std::max(1, std::min(k_semantic / 8, static_cast<int>(center_scores.size())));
            std::unordered_set<int32_t> selected_centers;
            std::vector<int32_t> selected_centers_vec;
            for (int idx = 0; idx < k_centers; ++idx) {
                selected_centers.insert(center_scores[idx].second);
                selected_centers_vec.push_back(center_scores[idx].second);
            }
            
            // 2. Gather slots associated with these centers, grouped by concentric role (Center, Around, Outer)
            // 2. Gather slots associated with these centers, grouped by concentric role (Center, Around, Outer) and center association
            // We use centers order from selected_centers_vec.
            std::vector<int32_t> centers_order = selected_centers_vec;
            // Map: center_id -> list of around/outer slots
            std::unordered_map<int32_t, std::vector<int32_t>> around_by_center;
            std::unordered_map<int32_t, std::vector<int32_t>> outer_by_center;
            
            const auto& role_map = cg.role_mapping_tensor;
            const auto& slot_to_center = cg.slot_to_center_tensor;
            
            for (int32_t s : srl_state.semantic_index.slot_ids) {
                if (selected_centers.count(s)) continue;
                
                if (s >= 0 && s < static_cast<int32_t>(slot_to_center.size())) {
                    int32_t assoc_c = slot_to_center[s];
                    if (selected_centers.count(assoc_c)) {
                        int role = role_map[s];
                        if (role == 1) {
                            around_by_center[assoc_c].push_back(s);
                        } else if (role == 0) {
                            outer_by_center[assoc_c].push_back(s);
                        }
                    }
                }
            }
            
            // Interleave around slots
            std::vector<int32_t> around_slots_interleaved;
            size_t max_len_around = 0;
            for (int32_t c : centers_order) {
                max_len_around = std::max(max_len_around, around_by_center[c].size());
            }
            for (size_t step = 0; step < max_len_around; ++step) {
                for (int32_t c : centers_order) {
                    if (step < around_by_center[c].size()) {
                        around_slots_interleaved.push_back(around_by_center[c][step]);
                    }
                }
            }
            
            // Interleave outer slots
            std::vector<int32_t> outer_slots_interleaved;
            size_t max_len_outer = 0;
            for (int32_t c : centers_order) {
                max_len_outer = std::max(max_len_outer, outer_by_center[c].size());
            }
            for (size_t step = 0; step < max_len_outer; ++step) {
                for (int32_t c : centers_order) {
                    if (step < outer_by_center[c].size()) {
                        outer_slots_interleaved.push_back(outer_by_center[c][step]);
                    }
                }
            }
            
            // 3. Build prioritized list: Center -> Around -> Outer (interleaved)
            std::vector<int32_t> prioritized_slots;
            std::unordered_set<int32_t> seen_prioritized;
            
            auto add_to_pri = [&](int32_t slot) {
                if (seen_prioritized.find(slot) == seen_prioritized.end()) {
                    seen_prioritized.insert(slot);
                    prioritized_slots.push_back(slot);
                }
            };
            
            for (int32_t c : centers_order) {
                add_to_pri(c);
            }
            for (int32_t s : around_slots_interleaved) {
                add_to_pri(s);
            }
            for (int32_t s : outer_slots_interleaved) {
                add_to_pri(s);
            }
            
            sem_slots = prioritized_slots;
            if (static_cast<int>(sem_slots.size()) > k_semantic) {
                sem_slots.resize(k_semantic);
            }
            concentric_routed = true;
        }
    }

    if (!concentric_routed) {
        if (!srl_state.chunk_graph.parent_landmarks.empty() && 
            !srl_state.chunk_graph.parent_to_children_tensor.empty()) {
            
            // 1. Collect unique parent landmarks
            std::vector<int32_t> parent_slots;
            std::vector<int> parent_rows;
            std::unordered_set<int32_t> seen_parents;
            
            for (int32_t slot : srl_state.chunk_graph.parent_landmarks) {
                if (slot >= 0 && seen_parents.find(slot) == seen_parents.end()) {
                    int row = srl_state.semantic_index.slot_to_idx(slot);
                    if (row >= 0) {
                        seen_parents.insert(slot);
                        parent_slots.push_back(slot);
                        parent_rows.push_back(row);
                    }
                }
            }
            
            if (!parent_rows.empty()) {
                // Score parent landmark blocks using dot product
                std::vector<std::pair<float, int32_t>> parent_scores;
                parent_scores.reserve(parent_rows.size());
                
                for (size_t idx = 0; idx < parent_rows.size(); ++idx) {
                    int row = parent_rows[idx];
                    int32_t slot = parent_slots[idx];
                    
                    float dot = 0.0f;
                    const float* desc = srl_state.semantic_index.desc_matrix.data() + (size_t)row * DESC_DIM;
                    for (int d = 0; d < DESC_DIM; ++d) {
                        dot += desc[d] * q_desc[d];
                    }
                    parent_scores.push_back({dot, slot});
                }
                
                // Sort parent scores descending
                std::sort(parent_scores.begin(), parent_scores.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });
                
                best_sem_score = parent_scores[0].first;
                
                // Select top landmark parents
                int k_parent = std::max(1, std::min(k_semantic / 8, static_cast<int>(parent_scores.size())));
                std::vector<int32_t> selected_parents;
                for (int idx = 0; idx < k_parent; ++idx) {
                    selected_parents.push_back(parent_scores[idx].second);
                }
                
                // 2. Gather children blocks per parent landmark and interleave
                int max_children = srl_state.chunk_graph.max_children;
                const auto& ptc = srl_state.chunk_graph.parent_to_children_tensor;
                int max_ptc_slots = ptc.empty() ? 0 : (ptc.size() / max_children);
                
                std::vector<std::vector<int32_t>> children_lists(selected_parents.size());
                size_t max_children_len = 0;
                
                for (size_t i = 0; i < selected_parents.size(); ++i) {
                    int32_t parent = selected_parents[i];
                    if (parent >= 0 && parent < max_ptc_slots) {
                        int base_idx = parent * max_children;
                        for (int c = 0; c < max_children; ++c) {
                            int32_t child = ptc[base_idx + c];
                            if (child != -1) {
                                children_lists[i].push_back(child);
                            }
                        }
                    }
                    max_children_len = std::max(max_children_len, children_lists[i].size());
                }
                
                std::vector<int32_t> children_interleaved;
                for (size_t step = 0; step < max_children_len; ++step) {
                    for (size_t i = 0; i < selected_parents.size(); ++i) {
                        if (step < children_lists[i].size()) {
                            children_interleaved.push_back(children_lists[i][step]);
                        }
                    }
                }
                
                std::vector<int32_t> hierarchical_slots;
                for (int32_t parent : selected_parents) {
                    hierarchical_slots.push_back(parent);
                }
                hierarchical_slots.insert(hierarchical_slots.end(), children_interleaved.begin(), children_interleaved.end());
                
                // Deduplicate preserving order
                std::unordered_set<int32_t> seen_hier;
                for (int32_t slot : hierarchical_slots) {
                    if (seen_hier.find(slot) == seen_hier.end()) {
                        seen_hier.insert(slot);
                        sem_slots.push_back(slot);
                    }
                }
                
                // Limit to k_semantic
                if (static_cast<int>(sem_slots.size()) > k_semantic) {
                    sem_slots.resize(k_semantic);
                }
            } else {
                // Fallback to standard semantic search if no parent landmarks are active
                auto sem_results = srl_state.semantic_index.search_with_scores(q_desc.data(), k_semantic);
                best_sem_score = sem_results.empty() ? 0.0f : sem_results[0].second;
                for (auto& [slot, score] : sem_results) sem_slots.push_back(slot);
            }
        } else {
            // Fallback to standard semantic search
            auto sem_results = srl_state.semantic_index.search_with_scores(q_desc.data(), k_semantic);
            best_sem_score = sem_results.empty() ? 0.0f : sem_results[0].second;
            for (auto& [slot, score] : sem_results) sem_slots.push_back(slot);
        }
    }

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

        // Determine seeds (semantic + lexical/rare lexical if not topic switch)
        std::vector<int32_t> seeds;
        int n_sem_seeds = std::max(1, k_semantic);
        int n_rare_seeds = std::max(2, (int)(2 * k_lexical));
        int n_lex_seeds = std::max(1, k_lexical);

        std::vector<int32_t> top_sem_slots;
        for (int i = 0; i < std::min(n_sem_seeds, (int)sem_slots.size()); ++i) {
            top_sem_slots.push_back(sem_slots[i]);
        }

        bool is_topic_switch = (best_sem_score < 0.25f);
        std::unordered_set<int32_t> seed_set;
        auto add_seed = [&](int32_t slot) {
            if (seed_set.find(slot) == seed_set.end()) {
                seed_set.insert(slot);
                seeds.push_back(slot);
            }
        };

        for (int32_t s : top_sem_slots) {
            add_seed(s);
        }

        if (!is_topic_switch) {
            for (int i = 0; i < std::min(n_rare_seeds, (int)rare_lex_slots.size()); ++i) {
                add_seed(rare_lex_slots[i]);
            }
            for (int i = 0; i < std::min(n_lex_seeds, (int)lex_slots.size()); ++i) {
                add_seed(lex_slots[i]);
            }
        }

        std::vector<float> A0_1(N, 0.0f);
        std::vector<float> A0_2(N, 0.0f);
        std::vector<float> A0_0(N, 0.0f);

        int q_seg = srl_state.current_query_segment_id;

        for (int i = 0; i < N; ++i) {
            int32_t slot = sem_idx.slot_ids[i];
            if (seed_set.count(slot)) {
                int seg = 0;
                auto it = srl_state.segment_ids.find(slot);
                if (it != srl_state.segment_ids.end()) {
                    seg = it->second;
                }
                if (seg == 1) {
                    if (q_seg != 2) { // ignore Segment 1 seeds if q_seg is 2
                        A0_1[i] = sem_scores_cpu[i];
                    }
                } else if (seg == 2) {
                    if (q_seg != 1) { // ignore Segment 2 seeds if q_seg is 1
                        A0_2[i] = sem_scores_cpu[i];
                    }
                } else {
                    A0_0[i] = sem_scores_cpu[i];
                }
            }
        }

        // retention[i] = graph_hop_decay * sem_scores_cpu[i]
        std::vector<float> retention(N);
        for (int i = 0; i < N; ++i)
            retention[i] = srl_state.graph_hop_decay * sem_scores_cpu[i];

        // 1. Walk for Segment 1: forbid Segment 2
        std::vector<float> A1_1 = graph_propagate(g, A0_1, retention, srl_state.graph_hop_decay, &sem_idx.slot_ids, &srl_state.slot_activation_strength);
        for (int i = 0; i < N; ++i) {
            int32_t slot = sem_idx.slot_ids[i];
            auto it = srl_state.segment_ids.find(slot);
            if (it != srl_state.segment_ids.end() && it->second == 2) {
                A1_1[i] = 0.0f;
            }
        }
        std::vector<float> A2_1 = graph_propagate(g, A1_1, retention, srl_state.graph_hop_decay, &sem_idx.slot_ids, &srl_state.slot_activation_strength);
        for (int i = 0; i < N; ++i) {
            int32_t slot = sem_idx.slot_ids[i];
            auto it = srl_state.segment_ids.find(slot);
            if (it != srl_state.segment_ids.end() && it->second == 2) {
                A2_1[i] = 0.0f;
            }
        }

        // 2. Walk for Segment 2: forbid Segment 1
        std::vector<float> A1_2 = graph_propagate(g, A0_2, retention, srl_state.graph_hop_decay, &sem_idx.slot_ids, &srl_state.slot_activation_strength);
        for (int i = 0; i < N; ++i) {
            int32_t slot = sem_idx.slot_ids[i];
            auto it = srl_state.segment_ids.find(slot);
            if (it != srl_state.segment_ids.end() && it->second == 1) {
                A1_2[i] = 0.0f;
            }
        }
        std::vector<float> A2_2 = graph_propagate(g, A1_2, retention, srl_state.graph_hop_decay, &sem_idx.slot_ids, &srl_state.slot_activation_strength);
        for (int i = 0; i < N; ++i) {
            int32_t slot = sem_idx.slot_ids[i];
            auto it = srl_state.segment_ids.find(slot);
            if (it != srl_state.segment_ids.end() && it->second == 1) {
                A2_2[i] = 0.0f;
            }
        }

        // 3. Walk for Segment 0: generic, forbid opponent segment dynamically
        std::vector<float> A1_0 = graph_propagate(g, A0_0, retention, srl_state.graph_hop_decay, &sem_idx.slot_ids, &srl_state.slot_activation_strength);
        if (q_seg == 1) {
            for (int i = 0; i < N; ++i) {
                int32_t slot = sem_idx.slot_ids[i];
                auto it = srl_state.segment_ids.find(slot);
                if (it != srl_state.segment_ids.end() && it->second == 2) {
                    A1_0[i] = 0.0f;
                }
            }
        } else if (q_seg == 2) {
            for (int i = 0; i < N; ++i) {
                int32_t slot = sem_idx.slot_ids[i];
                auto it = srl_state.segment_ids.find(slot);
                if (it != srl_state.segment_ids.end() && it->second == 1) {
                    A1_0[i] = 0.0f;
                }
            }
        }

        std::vector<float> A2_0 = graph_propagate(g, A1_0, retention, srl_state.graph_hop_decay, &sem_idx.slot_ids, &srl_state.slot_activation_strength);
        if (q_seg == 1) {
            for (int i = 0; i < N; ++i) {
                int32_t slot = sem_idx.slot_ids[i];
                auto it = srl_state.segment_ids.find(slot);
                if (it != srl_state.segment_ids.end() && it->second == 2) {
                    A2_0[i] = 0.0f;
                }
            }
        } else if (q_seg == 2) {
            for (int i = 0; i < N; ++i) {
                int32_t slot = sem_idx.slot_ids[i];
                auto it = srl_state.segment_ids.find(slot);
                if (it != srl_state.segment_ids.end() && it->second == 1) {
                    A2_0[i] = 0.0f;
                }
            }
        }

        // Sum results
        std::vector<float> A1(N);
        std::vector<float> A2(N);
        for (int i = 0; i < N; ++i) {
            A1[i] = A1_1[i] + A1_2[i] + A1_0[i];
            A2[i] = A2_1[i] + A2_2[i] + A2_0[i];
        }

        // graph_scores = A1 + A2, exclude seed rows
        std::vector<std::pair<float, int32_t>> gscore_slots;
        for (int i = 0; i < N; ++i) {
            if (seed_set.count(sem_idx.slot_ids[i])) continue;
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
    // Step 7.5: Dynamic Anchors expansion
    // -------------------------------------------------------------------
    std::vector<int32_t> dynamic_routed_slots;
    if (!srl_state.dynamic_anchors.empty()) {
        std::unordered_set<int32_t> da_set(srl_state.dynamic_anchors.begin(), srl_state.dynamic_anchors.end());
        dynamic_routed_slots = srl_state.expand_neighborhood(da_set);
    }

    // -------------------------------------------------------------------
    // Step 7.6: Prompt Anchors expansion
    // -------------------------------------------------------------------
    std::vector<int32_t> prompt_routed_slots;
    if (!srl_state.prompt_anchors.empty()) {
        int b_size = srl_state.inverted_index.block_size;
        std::unordered_set<int32_t> pa_slots;
        for (int idx : srl_state.prompt_anchors) {
            int block_idx = idx / b_size;
            if (block_idx >= 0 && block_idx < static_cast<int>(srl_state.ordered_slot_ids.size())) {
                pa_slots.insert(srl_state.ordered_slot_ids[block_idx]);
            }
        }
        if (!pa_slots.empty()) {
            prompt_routed_slots = srl_state.expand_neighborhood(pa_slots);
        }
    }

    // -------------------------------------------------------------------
    // Step 8: Merge and deduplicate
    // Order: sink > semantic > rare_lex > graph > lexical > recency > dynamic_routed > prompt_routed
    // -------------------------------------------------------------------
    std::unordered_set<int32_t> seen;
    std::vector<int32_t> combined;

    auto add_unique = [&](const std::vector<int32_t>& src) {
        for (int32_t s : src) {
            if (!seen.count(s)) {
                seen.insert(s);
                combined.push_back(s);
            }
        }
    };

    add_unique(srl_state.sink_blocks);
    add_unique(sem_slots);
    add_unique(rare_lex_slots);
    add_unique(graph_slots);
    add_unique(lex_slots);
    add_unique(recency_slots);
    add_unique(dynamic_routed_slots);
    add_unique(prompt_routed_slots);

    // -------------------------------------------------------------------
    // Step 8.5: Structured Attention Segmenting filtering
    // -------------------------------------------------------------------
    int curr_seg = srl_state.current_query_segment_id;
    if (curr_seg != 0 && !srl_state.segment_ids.empty()) {
        std::unordered_set<int32_t> sink_set(srl_state.sink_blocks.begin(), srl_state.sink_blocks.end());
        std::vector<int32_t> filtered;
        for (int32_t slot : combined) {
            if (sink_set.count(slot)) {
                filtered.push_back(slot);
                continue;
            }
            auto it = srl_state.segment_ids.find(slot);
            if (it != srl_state.segment_ids.end()) {
                int seg_id = it->second;
                if (seg_id == 0 || seg_id == curr_seg) {
                    filtered.push_back(slot);
                }
            } else {
                filtered.push_back(slot);
            }
        }
        combined = filtered;
    }

    if (curr_seg == 0 && N - static_cast<int>(combined.size()) <= 2) {
        combined = srl_state.ordered_slot_ids;
    }

    if (combined.empty()) {
        combined = srl_state.ordered_slot_ids;
        if (static_cast<int>(combined.size()) > K) {
            combined.resize(K);
        }
    }

    // -------------------------------------------------------------------
    // Step 9: Two-level gate: rerank non-sink candidates
    // -------------------------------------------------------------------
    std::unordered_set<int32_t> sink_set(srl_state.sink_blocks.begin(), srl_state.sink_blocks.end());
    std::vector<int32_t> sink_candidates;
    std::vector<int32_t> non_sink_candidates;
    for (int32_t s : combined) {
        if (sink_set.count(s)) {
            sink_candidates.push_back(s);
        } else {
            non_sink_candidates.push_back(s);
        }
    }

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

inline std::vector<int32_t> route_query_fixed_k(
    const float*            Q,           // [H * D]
    int                     H,
    int                     D,
    SessionSRLState&        srl_state,
    const BlockPoolInterface& pool,
    float                   scale,
    int                     layer_idx,
    const std::vector<int>* query_tokens = nullptr
) {
    std::vector<int32_t> selected = route_query(Q, H, D, srl_state, pool, scale, layer_idx, query_tokens);
    int k_fixed = 64;
    if (const char* env_k = std::getenv("DIFFKV_SRL_K_FIXED")) {
        k_fixed = std::atoi(env_k);
    }
    
    if (selected.empty()) {
        return std::vector<int32_t>(k_fixed, 0);
    }
    
    if ((int)selected.size() >= k_fixed) {
        selected.resize(k_fixed);
        return selected;
    }
    
    int pad_count = k_fixed - selected.size();
    int32_t last_val = selected.back();
    for (int i = 0; i < pad_count; ++i) {
        selected.push_back(last_val);
    }
    return selected;
}

// Out-of-line SRL state builders/helpers defined in query_router.cpp
void update_srl_from_compressed_block(
    SessionSRLState&               state,
    const float*                   desc_f32,
    int32_t                        slot_id,
    const int32_t*                 token_ids,
    int                            block_len,
    int                            start_pos,
    const std::unordered_set<int>& stop_tokens
);

SessionSRLState build_srl_state_from_blocks(
    const float*                   desc_matrix,
    const int32_t*                 slot_ids,
    int                            N,
    const int32_t*                 token_ids,
    int                            seq_len,
    int                            block_size,
    const std::unordered_set<int>& stop_tokens,
    int                            K_semantic         = 6,
    int                            K_temporal         = 2,
    float                          overlap_threshold  = 0.15f,
    bool                           add_first_as_sink  = true,
    bool                           add_last_as_sink   = true,
    const std::vector<int>*        block_anchor_idxs  = nullptr,
    int                            cached_len         = 0
);

std::string format_routing_stats(const SessionSRLState& state);

} // namespace diffkv
