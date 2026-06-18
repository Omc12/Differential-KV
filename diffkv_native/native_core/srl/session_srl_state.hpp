// session_srl_state.hpp
// Translation of session_srl_state.py to C++17.
// Maintains all per-session SRL (Semantic Routing Layer) state.

#pragma once

#include "native_core/srl/semantic_index.hpp"
#include "native_core/srl/chunk_graph.hpp"
#include "native_core/srl/inverted_index.hpp"
#include "native_core/srl/factual_store.hpp"

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <numeric>
#include <functional>

namespace diffkv {

struct SessionSRLStateHistoryEntry {
    std::vector<std::vector<int32_t>> vsl_active_candidates;
    int32_t vsl_consecutive_helpers = 0;
    int32_t current_entity_id = -1;
    std::unordered_set<int32_t> current_step_factual_tokens;
    std::vector<std::vector<int32_t>> current_step_factual_sequences;
    std::vector<int32_t> current_step_sequence_entity_ids;
    std::vector<bool> current_step_sequence_is_prime;
    std::vector<std::vector<int32_t>> current_step_sequence_prefixes;
    std::vector<int32_t> current_step_slots;
    float current_step_max_similarity = 0.0f;
    std::vector<int32_t> dynamic_anchors;
    std::vector<int32_t> generated_token_slots;
    std::vector<int> recent_generated_tokens;
    std::vector<std::vector<float>> recent_decode_keys;
    int current_step_count = 0;
};

// ---------------------------------------------------------------------------
// SessionSRLState
//
// Holds all mutable SRL state for one active decode session.
// ---------------------------------------------------------------------------
struct SessionSRLState {
    // --- Core retrieval structures ---
    SemanticIndex      semantic_index;
    ChunkGraph         chunk_graph;
    InvertedTokenIndex inverted_index;
    FactualExactStore  factual_store;

    // --- Block ordering ---
    std::vector<int32_t> ordered_slot_ids;   // pool slot IDs in chronological order
    std::vector<int32_t> sink_blocks;        // always-included (e.g., first/last blocks)

    // --- Adaptive-K running state ---
    float recent_miss_rate = 0.0f;
    float k_multiplier     = 1.0f;
    int   call_count       = 0;

    // --- Token tracking ---
    std::vector<int> recent_generated_tokens;  // rolling window of generated token IDs
    std::vector<int> current_query_tokens;     // current query token IDs
    std::vector<int> ordered_anchor_idxs;      // anchor token indices in chronological order
    int cached_len = 0;                        // cached sequence length (turn boundary)

    // --- Per-decode-step cached slot selection ---
    // Mirrors ACTIVE_RUNTIME/runtime/diffkv_attention.py:544
    //   srl_state.current_step_slots = selected_slots  (set at layer 0)
    //   selected_slots = srl_state.current_step_slots  (reused for layers 1-N)
    // In diffkv_native this is populated by route_decode_slots() and read
    // back by any subsystem that needs the current routing result.
    std::vector<int32_t> current_step_slots;
    int current_step_count = 0;
    int current_step_step  = -1;  // decode step index for which current_step_slots is valid

    // Clear per-step routing cache (call at the start of each decode step,
    // mirroring Python's implicit invalidation before layer-0 re-routes).
    void clear_step_cache() {
        current_step_slots.clear();
        current_step_step = -1;
    }

    // --- Chunk graph rebuild throttle ---
    // Mirrors ACTIVE_RUNTIME batch_engine.py:1034:
    //   n_at_build = getattr(srl_state_existing, "n_blocks_at_build", n_current)
    //   growth_ratio = (n_current - n_at_build) / max(1, n_at_build)
    //   if growth_ratio < 0.20: skip rebuild
    // Tracks how many blocks were present the last time chunk_graph was built.
    int n_blocks_at_last_graph_build = 0;

    // --- Configuration knobs ---
    int   k_min              = 20;
    int   k_max              = 200;
    float k_semantic_frac    = 0.50f;
    float k_lexical_frac     = 0.15f;
    float k_graph_frac       = 0.15f;
    float k_recency_frac     = 0.20f;
    int   routing_threshold  = 2;    // min blocks before SRL activates
    float overlap_threshold  = 0.15f;
    float graph_hop_decay    = 0.5f;
    float srl_age_penalty    = 0.01f;

    // -----------------------------------------------------------------
    // n_active_blocks: number of blocks currently tracked
    // -----------------------------------------------------------------
    int n_active_blocks() const {
        return static_cast<int>(ordered_slot_ids.size());
    }

    // -----------------------------------------------------------------
    // update_generated_tokens
    // Append a new token to the rolling window, capping to maxlen.
    // -----------------------------------------------------------------
    void update_generated_tokens(int token_id, int maxlen = 64) {
        recent_generated_tokens.push_back(token_id);
        if (static_cast<int>(recent_generated_tokens.size()) > maxlen) {
            int excess = static_cast<int>(recent_generated_tokens.size()) - maxlen;
            recent_generated_tokens.erase(
                recent_generated_tokens.begin(),
                recent_generated_tokens.begin() + excess);
        }
    }

    // -----------------------------------------------------------------
    // update_miss_rate
    //
    // Update the EMA miss-rate signal based on the maximum attention
    // weight seen over the K retrieved blocks.
    //
    //   miss_signal = clamp(1 - (max_w - expected_max) / (1 - expected_max), 0, 1)
    //   recent_miss_rate = (1 - alpha) * recent_miss_rate + alpha * miss_signal
    //   then adjust k_multiplier accordingly.
    //
    // attn_weights: [K] float32 attention weights over retrieved blocks
    // -----------------------------------------------------------------
    void update_miss_rate(const float* attn_weights, int K) {
        if (K == 0) return;

        float max_w        = *std::max_element(attn_weights, attn_weights + K);
        float expected_max = 1.0f / static_cast<float>(K);
        float denom        = (1.0f - expected_max) + 1e-8f;
        float miss_signal  = std::max(0.0f, std::min(1.0f,
            1.0f - (max_w - expected_max) / denom));

        constexpr float alpha = 0.05f;
        recent_miss_rate = (1.0f - alpha) * recent_miss_rate + alpha * miss_signal;

        if (recent_miss_rate > 0.4f)
            k_multiplier = std::min(k_multiplier * 1.2f, 3.0f);
        else
            k_multiplier = std::max(k_multiplier * 0.99f, 1.0f);
    }

    std::unordered_set<int32_t> current_step_factual_tokens;
    std::vector<std::vector<int32_t>> current_step_factual_sequences;
    float current_step_max_similarity = 0.0f;
    // PERF: the factual store query is layer-independent (layer-0 K is the proxy).
    // Cache its result at layer 0 and reuse across layers 1..N-1 to avoid running
    // the query (+ deep K/V copies) once per layer per decode step.
    std::vector<FactEntry> step_cached_entries;
    std::vector<std::vector<int32_t>> vsl_active_candidates;
    int32_t vsl_consecutive_helpers = 0;
    // First decode-step layer-0 K stored as a query anchor.
    // Blended with current decode-K (65%/35%) on subsequent steps to prevent
    // semantic drift from pulling factual retrieval off the original topic.
    std::vector<float> factual_anchor_q;

    // Entity-subgraph tracking for relationship binding.
    // current_entity_id: document position (start_idx) of the active prime.
    //   -1 = no entity context yet. Persists across steps; reset at turn start.
    // current_step_sequence_entity_ids: parallel to current_step_factual_sequences;
    //   entity_id (prime start_idx) for each sequence (-1 = unknown).
    // current_step_sequence_is_prime: True if the sequence is itself a prime entry.
    int32_t current_entity_id = -1;
    // Dual-entity mode for comparison questions (e.g., "Compare EP2 and EP3").
    // When active, both entities' factual sequences are available for generation
    // but cross-entity contamination is prevented via strict VSL filtering.
    bool dual_entity_mode = false;
    std::vector<int32_t> dual_entity_ids;  // [entity_id_1, entity_id_2]
    // RC5 — explicit comparison mode: lock generation to one entity at a time
    // (per-entity blocks) instead of letting them interleave.
    std::vector<int32_t> comparison_entities;     // ordered block sequence
    int comparison_active_idx = 0;                // current block
    std::unordered_set<int32_t> comparison_covered;  // entities already produced
    std::vector<int32_t> current_step_sequence_entity_ids;
    std::vector<bool>    current_step_sequence_is_prime;
    // current_step_sequence_prefixes: parallel list; the source tokens preceding
    // each sequence's span (RC2 quote-grounded connectives). empty for triples.
    std::vector<std::vector<int32_t>> current_step_sequence_prefixes;

    // Segment variables for SAS
    std::unordered_map<int32_t, int> segment_ids; // maps slot_id -> segment_id
    int current_query_segment_id = 0;
    int concept_tok_1 = -1;
    int concept_tok_2 = -1;

    // EQA-DR variables
    std::vector<float> prompt_eagle_scores;
    std::vector<int> prompt_anchors;
    std::vector<std::vector<float>> recent_decode_keys;
    std::vector<int32_t> generated_token_slots;
    std::vector<int32_t> dynamic_anchors;

    // Dynamic Node Reinforcement Strength
    std::unordered_map<int32_t, float> slot_activation_strength;

    std::vector<int32_t> expand_neighborhood(const std::unordered_set<int32_t>& seed_slots) const {
        std::unordered_set<int32_t> expanded(seed_slots.begin(), seed_slots.end());
        if (chunk_graph.neighbors.empty() || semantic_index.slot_ids.empty() || chunk_graph.max_degree <= 0) {
            return std::vector<int32_t>(expanded.begin(), expanded.end());
        }

        // Map slot_ids to row indices for fast lookup
        std::unordered_map<int32_t, int> slot_to_row;
        for (size_t i = 0; i < semantic_index.slot_ids.size(); ++i) {
            slot_to_row[semantic_index.slot_ids[i]] = static_cast<int>(i);
        }

        int N_neighbors = chunk_graph.max_degree;

        for (int32_t slot : seed_slots) {
            auto it = slot_to_row.find(slot);
            if (it != slot_to_row.end()) {
                int row_idx = it->second;
                int start_nb = row_idx * N_neighbors;
                if (start_nb + N_neighbors <= static_cast<int>(chunk_graph.neighbors.size())) {
                    for (int nb = 0; nb < N_neighbors; ++nb) {
                        int32_t nb_row = chunk_graph.neighbors[start_nb + nb];
                        if (nb_row >= 0 && nb_row < static_cast<int32_t>(semantic_index.slot_ids.size())) {
                            expanded.insert(semantic_index.slot_ids[nb_row]);
                        }
                    }
                }
            }
        }
        return std::vector<int32_t>(expanded.begin(), expanded.end());
    }

    void update_query_segment(int token_id) {
        // Look at the last 12 generated tokens
        int size = static_cast<int>(recent_generated_tokens.size());
        int start = std::max(0, size - 12);
        
        float seg1_score = 0.0f;
        float seg2_score = 0.0f;
        
        for (int i = start; i < size; ++i) {
            int tid = recent_generated_tokens[i];
            auto it = inverted_index.occurrences.find(tid);
            if (it != inverted_index.occurrences.end()) {
                const auto& occs = it->second;
                float total_cnt = static_cast<float>(occs.size());
                if (total_cnt > 0.0f) {
                    float seg1_cnt = 0.0f;
                    float seg2_cnt = 0.0f;
                    for (const auto& occ : occs) {
                        int32_t slot = std::get<0>(occ);
                        auto seg_it = segment_ids.find(slot);
                        if (seg_it != segment_ids.end()) {
                            int seg = seg_it->second;
                            if (seg == 1) {
                                seg1_cnt += 1.0f;
                            } else if (seg == 2) {
                                seg2_cnt += 1.0f;
                            }
                        }
                    }
                    float idf_val = 1.0f;
                    auto idf_it = inverted_index.idf.find(tid);
                    if (idf_it != inverted_index.idf.end()) {
                        idf_val = idf_it->second;
                    }
                    seg1_score += (seg1_cnt / total_cnt) * idf_val;
                    seg2_score += (seg2_cnt / total_cnt) * idf_val;
                }
            }
        }
        
        if (seg1_score > seg2_score + 1.0f) {
            current_query_segment_id = 1;
        } else if (seg2_score > seg1_score + 1.0f) {
            current_query_segment_id = 2;
        } else {
            current_query_segment_id = 0;
        }
    }

    void update_dynamic_anchors(const std::unordered_set<int>& stop_token_ids) {
        int L_g = recent_decode_keys.size();
        if (L_g < 2 || L_g % 4 != 0) return;

        // 1. Compute self-similarity matrix
        int head_dim = recent_decode_keys[0].size();
        std::vector<float> sim(L_g * L_g, 0.0f);
        for (int i = 0; i < L_g; ++i) {
            for (int j = 0; j < L_g; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += recent_decode_keys[i][d] * recent_decode_keys[j][d];
                }
                sim[i * L_g + j] = dot / std::sqrt(head_dim);
            }
        }

        // 2. Apply causal mask and softmax
        for (int i = 0; i < L_g; ++i) {
            float max_val = -1e9f;
            for (int j = 0; j < L_g; ++j) {
                if (j >= i) {
                    sim[i * L_g + j] = -1e9f;
                }
                if (sim[i * L_g + j] > max_val) {
                    max_val = sim[i * L_g + j];
                }
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < L_g; ++j) {
                sim[i * L_g + j] = std::exp(sim[i * L_g + j] - max_val);
                sum_exp += sim[i * L_g + j];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-10f);
            for (int j = 0; j < L_g; ++j) {
                sim[i * L_g + j] *= inv_sum;
            }
        }

        // 3. Compute lookback score R_gen
        std::vector<float> R_gen(L_g, 0.0f);
        for (int j = 0; j < L_g; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < L_g; ++i) {
                col_sum += sim[i * L_g + j];
            }
            R_gen[j] = col_sum;
        }

        // 4. Update dynamic anchors
        dynamic_anchors.clear();
        for (int i = 0; i < L_g; ++i) {
            if (R_gen[i] > 1.5f) {
                // Verify it's not a stop token
                if (i < static_cast<int>(recent_generated_tokens.size())) {
                    int tid = recent_generated_tokens[i];
                    if (stop_token_ids.find(tid) == stop_token_ids.end()) {
                        if (i < static_cast<int>(generated_token_slots.size())) {
                            dynamic_anchors.push_back(generated_token_slots[i]);
                        }
                    }
                }
            }
        }
    }

    void setup_sas_and_eqa(const std::vector<int32_t>& token_ids,
                           const std::unordered_set<int>& stop_token_ids,
                           std::function<std::string(int32_t)> token_to_piece_fn = nullptr) {
        prompt_eagle_scores = factual_store.eagle_scores;
        if (prompt_eagle_scores.empty() || token_ids.empty()) {
            return;
        }

        int L = static_cast<int>(token_ids.size());
        std::vector<std::pair<int, float>> valid_candidates;
        for (int i = 0; i < L; ++i) {
            int32_t tid = token_ids[i];
            if (stop_token_ids.find(tid) == stop_token_ids.end()) {
                float score = prompt_eagle_scores[i];
                
                // Boost known comparative words
                if (token_to_piece_fn) {
                    std::string word = token_to_piece_fn(tid);
                    // trim word
                    word.erase(0, word.find_first_not_of(" \t\r\n"));
                    word.erase(word.find_last_not_of(" \t\r\n") + 1);
                    std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return std::tolower(c); });
                    
                    // F20: generalised salience boost — replaces the benchmark-overfit
                    // physics/math word list (tech-debt #2) with a domain-agnostic rule.
                    // Numeric tokens are universally salient for factual retrieval (codes,
                    // IDs, quantities, dates). Mirrors ACTIVE_RUNTIME session_srl_state.py.
                    bool is_numeric = !word.empty();
                    for (char c : word) if (!std::isdigit(static_cast<unsigned char>(c))) { is_numeric = false; break; }
                    if (is_numeric) {
                        score += 5.0f;
                    }
                }
                
                valid_candidates.push_back({i, score});
            }
        }

        // Sort by score descending
        std::sort(valid_candidates.begin(), valid_candidates.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        prompt_anchors.clear();
        for (size_t i = 0; i < std::min(valid_candidates.size(), size_t(3)); ++i) {
            prompt_anchors.push_back(valid_candidates[i].first);
        }

        // Extract unique candidate token IDs
        std::vector<int32_t> candidate_tids;
        for (const auto& cand : valid_candidates) {
            int32_t tid = token_ids[cand.first];
            if (std::find(candidate_tids.begin(), candidate_tids.end(), tid) == candidate_tids.end()) {
                candidate_tids.push_back(tid);
            }
        }

        // Helpers for Jaccard overlap
        auto get_slots = [&](int32_t tid) -> std::unordered_set<int32_t> {
            std::unordered_set<int32_t> slots;
            auto it = inverted_index.occurrences.find(tid);
            if (it != inverted_index.occurrences.end()) {
                for (const auto& occ : it->second) {
                    slots.insert(std::get<0>(occ));
                }
            }
            return slots;
        };

        auto jaccard = [](const std::unordered_set<int32_t>& s1, const std::unordered_set<int32_t>& s2) -> float {
            if (s1.empty() || s2.empty()) return 0.0f;
            int intersect_cnt = 0;
            for (int32_t x : s1) {
                if (s2.count(x)) intersect_cnt++;
            }
            int union_cnt = s1.size() + s2.size() - intersect_cnt;
            return static_cast<float>(intersect_cnt) / union_cnt;
        };

        int c_tok_1 = concept_tok_1;
        int c_tok_2 = concept_tok_2;

        if (c_tok_1 == -1) {
            int first_idx = -1;
            for (size_t i = 0; i < candidate_tids.size(); ++i) {
                auto s = get_slots(candidate_tids[i]);
                if (!s.empty()) {
                    c_tok_1 = candidate_tids[i];
                    first_idx = static_cast<int>(i);
                    break;
                }
            }

            if (first_idx != -1) {
                auto s1 = get_slots(c_tok_1);
                for (size_t j = first_idx + 1; j < candidate_tids.size(); ++j) {
                    int32_t tid2 = candidate_tids[j];
                    auto s2 = get_slots(tid2);
                    if (!s2.empty()) {
                        float j_val = jaccard(s1, s2);
                        if (j_val <= 0.2f) {
                            c_tok_2 = tid2;
                            break;
                        }
                    }
                }
                if (c_tok_2 == -1) {
                    // Fallback to minimum Jaccard overlap
                    float min_j = 1.0f;
                    int32_t best_tid2 = -1;
                    for (size_t j = first_idx + 1; j < candidate_tids.size(); ++j) {
                        int32_t tid2 = candidate_tids[j];
                        auto s2 = get_slots(tid2);
                        if (!s2.empty()) {
                            float j_val = jaccard(s1, s2);
                            if (j_val < min_j) {
                                min_j = j_val;
                                best_tid2 = tid2;
                            }
                        }
                    }
                    c_tok_2 = best_tid2;
                }
            }
            concept_tok_1 = c_tok_1;
            concept_tok_2 = c_tok_2;
        }

        // Map segments preserving existing ones
        std::unordered_map<int32_t, int> existing_segs = segment_ids;
        segment_ids.clear();
        for (int32_t slot : ordered_slot_ids) {
            auto it = existing_segs.find(slot);
            segment_ids[slot] = (it != existing_segs.end()) ? it->second : 0;
        }

        std::unordered_set<int32_t> slots_1;
        std::unordered_set<int32_t> slots_2;

        if (concept_tok_1 != -1) {
            slots_1 = get_slots(concept_tok_1);
        }
        if (concept_tok_2 != -1) {
            slots_2 = get_slots(concept_tok_2);
        }

        // Neighborhood Expansion
        std::vector<int32_t> expanded_1_vec = expand_neighborhood(slots_1);
        std::vector<int32_t> expanded_2_vec = expand_neighborhood(slots_2);

        std::unordered_set<int32_t> expanded_1(expanded_1_vec.begin(), expanded_1_vec.end());
        std::unordered_set<int32_t> expanded_2(expanded_2_vec.begin(), expanded_2_vec.end());

        for (int32_t slot : ordered_slot_ids) {
            auto it = existing_segs.find(slot);
            if (it != existing_segs.end() && it->second != 0) {
                continue;
            }
            if (cached_len > 0) {
                segment_ids[slot] = 0;
                continue;
            }
            bool in_1 = expanded_1.find(slot) != expanded_1.end();
            bool in_2 = expanded_2.find(slot) != expanded_2.end();
            if (in_1 && in_2) {
                segment_ids[slot] = 0;
            } else if (in_1) {
                segment_ids[slot] = 1;
            } else if (in_2) {
                segment_ids[slot] = 2;
            } else {
                segment_ids[slot] = 0;
            }
        }
    }

    // -----------------------------------------------------------------
    // reset_step_cache
    // Called at the start of each new decode step to clear the cached
    // slot selection from the previous step.
    // -----------------------------------------------------------------
    void reset_step_cache() {
        current_step_slots.clear();
        current_step_count = 0;
        current_step_factual_tokens.clear();
        current_step_factual_sequences.clear();
        current_step_max_similarity = 0.0f;
    }

    // -----------------------------------------------------------------
    // is_active: returns true when enough blocks exist to engage SRL
    // -----------------------------------------------------------------
    bool is_active() const {
        return n_active_blocks() >= routing_threshold;
    }

    // -----------------------------------------------------------------
    // k_components: decompose K budget into per-channel counts
    // -----------------------------------------------------------------
    struct KBudget {
        int k_semantic;
        int k_lexical;
        int k_graph;
        int k_recency;
    };

    KBudget k_components(int K) const {
        KBudget b;
        // RECONSTRUCTION FIX (F11/D7): ACTIVE_RUNTIME uses int(K*frac) truncation
        // (query_router.py k_semantic = max(1, int(K*_SEM_FRAC)) etc.), not round().
        b.k_semantic = static_cast<int>(K * k_semantic_frac);
        b.k_lexical  = static_cast<int>(K * k_lexical_frac);
        b.k_graph    = static_cast<int>(K * k_graph_frac);
        b.k_recency  = static_cast<int>(K * k_recency_frac);
        return b;
    }

    // -----------------------------------------------------------------
    // add_block: register a newly compressed block into all SRL structures.
    // desc: [DESC_DIM] float32, L2-normalised descriptor
    // slot_id: pool slot ID assigned to this block
    // -----------------------------------------------------------------
    void add_block(const float* desc, int32_t slot_id) {
        ordered_slot_ids.push_back(slot_id);
        add_block_to_index(semantic_index, desc, slot_id);
    }

    // --- Decode History & Rollback support ---
    std::unordered_map<int, SessionSRLStateHistoryEntry> step_history;
    mutable std::map<std::vector<int32_t>, const FactEntry*> entries_by_tokens_map;
    mutable std::vector<const FactEntry*> prime_entries;
    mutable std::unordered_map<size_t, int32_t> cached_triple_hash_to_entity;
    mutable std::unordered_map<int32_t, std::unordered_set<int32_t>> cached_prime_tokens_by_entity;
    mutable std::unordered_map<int32_t, std::unordered_set<int32_t>> cached_property_tokens_by_entity;
    mutable std::unordered_map<int32_t, std::vector<const FactEntry*>> entries_by_token_id;
    mutable bool entries_map_built = false;

    void save_step_state(int seq_len) {
        SessionSRLStateHistoryEntry entry;
        entry.vsl_active_candidates = vsl_active_candidates;
        entry.vsl_consecutive_helpers = vsl_consecutive_helpers;
        entry.current_entity_id = current_entity_id;
        entry.current_step_factual_tokens = current_step_factual_tokens;
        entry.current_step_factual_sequences = current_step_factual_sequences;
        entry.current_step_sequence_entity_ids = current_step_sequence_entity_ids;
        entry.current_step_sequence_is_prime = current_step_sequence_is_prime;
        entry.current_step_sequence_prefixes = current_step_sequence_prefixes;
        entry.current_step_slots = current_step_slots;
        entry.current_step_max_similarity = current_step_max_similarity;
        entry.dynamic_anchors = dynamic_anchors;
        entry.generated_token_slots = generated_token_slots;
        entry.recent_generated_tokens = recent_generated_tokens;
        entry.recent_decode_keys = recent_decode_keys;
        entry.current_step_count = current_step_count;
        
        step_history[seq_len] = std::move(entry);
    }

    void rollback_to(int target_len, const std::unordered_set<int32_t>* kept_slots = nullptr) {
        // 1. Filter slot-related lists if kept_slots is provided
        if (kept_slots != nullptr) {
            auto filter_func = [kept_slots](std::vector<int32_t>& vec) {
                std::vector<int32_t> filtered;
                for (int32_t slot : vec) {
                    if (kept_slots->count(slot)) {
                        filtered.push_back(slot);
                    }
                }
                vec = std::move(filtered);
            };
            filter_func(ordered_slot_ids);
            filter_func(sink_blocks);
            filter_func(dynamic_anchors);
            filter_func(generated_token_slots);
        }

        // 2. Restore decode state from step history
        auto it = step_history.find(target_len);
        if (it != step_history.end()) {
            const auto& entry = it->second;
            vsl_active_candidates = entry.vsl_active_candidates;
            vsl_consecutive_helpers = entry.vsl_consecutive_helpers;
            current_entity_id = entry.current_entity_id;
            current_step_factual_tokens = entry.current_step_factual_tokens;
            current_step_factual_sequences = entry.current_step_factual_sequences;
            current_step_sequence_entity_ids = entry.current_step_sequence_entity_ids;
            current_step_sequence_is_prime = entry.current_step_sequence_is_prime;
            current_step_sequence_prefixes = entry.current_step_sequence_prefixes;
            current_step_slots = entry.current_step_slots;
            current_step_max_similarity = entry.current_step_max_similarity;
            
            dynamic_anchors.clear();
            for (int32_t slot : entry.dynamic_anchors) {
                if (kept_slots == nullptr || kept_slots->count(slot)) {
                    dynamic_anchors.push_back(slot);
                }
            }
            
            generated_token_slots.clear();
            for (int32_t slot : entry.generated_token_slots) {
                if (kept_slots == nullptr || kept_slots->count(slot)) {
                    generated_token_slots.push_back(slot);
                }
            }
            
            recent_generated_tokens = entry.recent_generated_tokens;
            recent_decode_keys = entry.recent_decode_keys;
            current_step_count = entry.current_step_count;
        } else {
            vsl_active_candidates.clear();
            vsl_consecutive_helpers = 0;
            current_entity_id = -1;
            current_step_factual_tokens.clear();
            current_step_factual_sequences.clear();
            current_step_sequence_entity_ids.clear();
            current_step_sequence_is_prime.clear();
            current_step_sequence_prefixes.clear();
            current_step_slots.clear();
            current_step_max_similarity = 0.0f;
            dynamic_anchors.clear();
            generated_token_slots.clear();
            recent_generated_tokens.clear();
            recent_decode_keys.clear();
            current_step_count = 0;
        }

        // Clean up history entries greater than target_len
        std::vector<int> to_remove;
        for (const auto& pair : step_history) {
            if (pair.first > target_len) {
                to_remove.push_back(pair.first);
            }
        }
        for (int k : to_remove) {
            step_history.erase(k);
        }
    }
};

} // namespace diffkv
