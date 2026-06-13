#include "native_core/srl/factual_store.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace diffkv {

void FactualExactStore::build(
    const std::vector<std::vector<float>>& k_activations,
    const std::vector<std::vector<float>>& v_activations,
    const std::vector<int32_t>& token_ids,
    const float* W_proj,
    int desc_dim,
    int head_dim,
    int kv_heads,
    const std::unordered_set<int>& stop_token_ids,
    const std::vector<int32_t>& slot_ids,
    int block_size,
    const InvertedTokenIndex& inv_index,
    const std::unordered_set<int32_t>& semantic_prime_slots,
    bool use_salience_parser
) {
    int L = token_ids.size();
    if (L == 0 || k_activations.empty()) return;
    int num_layers = k_activations.size();
    int F_test = kv_heads * head_dim;
    this->num_layers = num_layers;
    this->F_test = F_test;

    std::vector<bool> factual_mask(L, false);

    if (use_salience_parser) {
        // 1. Compute Eagle lookback score R(t) using causal key self-similarity
        std::vector<float> R(L, 0.0f);
        if (L > 1) {
            std::vector<float> K_avg(L * head_dim, 0.0f);
            for (int t = 0; t < L; ++t) {
                for (int kh = 0; kh < kv_heads; ++kh) {
                    for (int d = 0; d < head_dim; ++d) {
                        K_avg[t * head_dim + d] += k_activations[0][t * F_test + kh * head_dim + d];
                    }
                }
                for (int d = 0; d < head_dim; ++d) {
                    K_avg[t * head_dim + d] /= kv_heads;
                }
            }

            std::vector<float> sim(L * L, 0.0f);
#ifdef __APPLE__
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, head_dim,
                        1.0f / std::sqrt(head_dim), K_avg.data(), head_dim,
                        K_avg.data(), head_dim, 0.0f, sim.data(), L);
#else
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += K_avg[i * head_dim + d] * K_avg[j * head_dim + d];
                    }
                    sim[i * L + j] = dot / std::sqrt(head_dim);
                }
            }
#endif

            // Apply causal mask and softmax along key dimensions
            for (int i = 0; i < L; ++i) {
                float max_val = -1e9f;
                for (int j = 0; j < L; ++j) {
                    if (j >= i) {
                        sim[i * L + j] = -1e9f;
                    }
                    if (sim[i * L + j] > max_val) {
                        max_val = sim[i * L + j];
                    }
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < L; ++j) {
                    sim[i * L + j] = std::exp(sim[i * L + j] - max_val);
                    sum_exp += sim[i * L + j];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (int j = 0; j < L; ++j) {
                    sim[i * L + j] *= inv_sum;
                }
            }

            // Sum columns to get total lookbacks pointing to each token
            for (int j = 0; j < L; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < L; ++i) {
                    col_sum += sim[i * L + j];
                }
                R[j] = col_sum;
            }
            this->eagle_scores = R;
        }

        // 2. Compute Key Norms at Layer 0
        std::vector<float> key_norms(L, 1.0f);
        for (int t = 0; t < L; ++t) {
            std::vector<float> K_avg_t(head_dim, 0.0f);
            for (int kh = 0; kh < kv_heads; ++kh) {
                for (int d = 0; d < head_dim; ++d) {
                    K_avg_t[d] += k_activations[0][t * F_test + kh * head_dim + d];
                }
            }
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                K_avg_t[d] /= kv_heads;
                sum_sq += K_avg_t[d] * K_avg_t[d];
            }
            key_norms[t] = std::sqrt(sum_sq);
        }

        // 3. Compute IDF values
        std::vector<float> idf_vals(L, 0.0f);
        for (int t = 0; t < L; ++t) {
            int tid = token_ids[t];
            auto it = inv_index.idf.find(tid);
            if (it != inv_index.idf.end()) {
                idf_vals[t] = it->second;
            } else {
                idf_vals[t] = (stop_token_ids.count(tid) ? 0.1f : 2.5f);
            }
        }

        // 4. Relational keyword IDF boost — give binding words (verbs,
        // prepositions, comparatives) a fixed IDF-equivalent score so they
        // survive the top-% salience selection.  Without this, they all score
        // ~0.1 and are systematically excluded, destroying relational structure.
        for (int t = 0; t < L; ++t) {
            int tid = token_ids[t];
            auto it = inv_index.idf.find(tid);
            if (it != inv_index.idf.end() && it->second < 1.5f) {
                // Low-IDF token — likely a relational/function word.
                // Boost to median content-word IDF so it has a chance to
                // survive salience selection when adjacent to concept tokens.
                idf_vals[t] = std::max(idf_vals[t], 2.0f);
            }
        }

        // 5. Compute joint factual salience score
        std::vector<float> total_salience(L, 0.0f);
        for (int t = 0; t < L; ++t) {
            total_salience[t] = key_norms[t] * idf_vals[t] * (1.0f + 1.0f * R[t]);
        }

        // Select the top 5% most salient tokens (precision mode: max 300 tokens)
        // 50% was selecting half the document — far too broad for exact grounding.
        // 5% keeps only the rarest, most distinctive content words per document.
        int k_num = std::max(8, (int)(L * 0.05f));
        k_num = std::min(k_num, 300);  // absolute cap regardless of document length
        k_num = std::min(k_num, L);

        if (L > 0) {
            std::vector<float> sorted_salience = total_salience;
            std::nth_element(sorted_salience.begin(), sorted_salience.end() - k_num, sorted_salience.end());
            float threshold_val = sorted_salience[L - k_num];
            for (int t = 0; t < L; ++t) {
                factual_mask[t] = (total_salience[t] >= threshold_val);
            }
        }

        // 5b. Relational Context Window Expansion — each salient seed token
        // is expanded into a ±3-token window so the factual span captures
        // the surrounding relational structure.
        {
            const int CONTEXT_WINDOW = 3;
            std::vector<bool> expanded_mask = factual_mask;
            for (int t = 0; t < L; ++t) {
                if (factual_mask[t]) {
                    int lo = std::max(0, t - CONTEXT_WINDOW);
                    int hi = std::min(L, t + CONTEXT_WINDOW + 1);
                    for (int p = lo; p < hi; ++p) {
                        expanded_mask[p] = true;
                    }
                }
            }
            factual_mask = expanded_mask;
        }

        // 5c. Gap-bridging (dilation)
        if (L > 2) {
            std::vector<bool> dilated_mask = factual_mask;
            for (int t = 1; t < L - 1; ++t) {
                if (factual_mask[t - 1] && factual_mask[t + 1]) {
                    dilated_mask[t] = true;
                }
            }
            factual_mask = dilated_mask;
        }
    } else {
        // Fallback to simple stop-token exclusion
        for (int t = 0; t < L; ++t) {
            int tid = token_ids[t];
            if (!stop_token_ids.count(tid) && tid > 0) {
                factual_mask[t] = true;
            }
        }
    }

    // 6. Group contiguous factual tokens into spans
    std::vector<std::pair<int, int>> spans;
    bool in_span = false;
    int start = -1;
    for (int t = 0; t < L; ++t) {
        if (factual_mask[t]) {
            if (!in_span) {
                start = t;
                in_span = true;
            }
        } else {
            if (in_span) {
                spans.push_back({start, t});
                in_span = false;
            }
        }
    }
    if (in_span) {
        spans.push_back({start, L});
    }

    // Split long spans into chunks of max length 20 (raised from 12).
    // Longer chunks preserve more relational context per sequence — critical for
    // VSL to lock onto full property phrases like "coalesce to a single direction
    // at rate sqrt(ε)" rather than truncated 12-token fragments.
    std::vector<std::pair<int, int>> chunked_spans;
    for (const auto& span : spans) {
        for (int sub_s = span.first; sub_s < span.second; sub_s += 20) {
            int sub_e = std::min(sub_s + 20, span.second);
            chunked_spans.push_back({sub_s, sub_e});
        }
    }

    // 7. Extract verbatim KV sequences across all layers for each span
    for (const auto& span : chunked_spans) {
        int s = span.first;
        int e = span.second;
        int span_len = e - s;
        if (span_len <= 0) continue;

        FactEntry entry;
        entry.start_idx = s;
        entry.end_idx = e;
        entry.K.resize(num_layers * F_test * span_len);
        entry.V.resize(num_layers * F_test * span_len);

        for (int l = 0; l < num_layers; ++l) {
            std::memcpy(entry.K.data() + l * F_test * span_len,
                        k_activations[l].data() + s * F_test,
                        span_len * F_test * sizeof(float));
            std::memcpy(entry.V.data() + l * F_test * span_len,
                        v_activations[l].data() + s * F_test,
                        span_len * F_test * sizeof(float));
        }

        // Compute descriptor for the span using layer 0 max-pooled key.
        // Max-pool over positions retains the most activated (distinctive) features
        // for each head, rather than averaging them away. This is critical for
        // formula/math spans where a single rare token dominates the span semantics.
        // Then mean over heads to produce the final descriptor vector.
        std::vector<float> max_k(head_dim, -1e9f);
        for (int t = 0; t < span_len; ++t) {
            for (int kh = 0; kh < kv_heads; ++kh) {
                for (int d = 0; d < head_dim; ++d) {
                    float val = entry.K[kh * head_dim + t * F_test + d];
                    if (val > max_k[d]) max_k[d] = val;
                }
            }
        }
        // Mean over heads by averaging max_k with itself (max is already position-pooled);
        // no further reduction needed — max_k[d] is already the per-position max.

        // Project descriptor using W_proj [desc_dim, head_dim] — using max_k
        entry.descriptor.resize(desc_dim, 0.0f);
        if (W_proj) {
            float norm_sq = 0.0f;
            for (int r = 0; r < desc_dim; ++r) {
                float val = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    val += max_k[d] * W_proj[r * head_dim + d];
                }
                entry.descriptor[r] = val;
                norm_sq += val * val;
            }
            float norm = std::sqrt(norm_sq) + 1e-8f;
            for (int r = 0; r < desc_dim; ++r) {
                entry.descriptor[r] /= norm;
            }
        }

        // Determine which slot IDs this span overlaps with
        if (block_size > 0) {
            int start_block_idx = s / block_size;
            int end_block_idx = (e - 1) / block_size;
            for (int idx = start_block_idx; idx <= end_block_idx; ++idx) {
                if (idx >= 0 && idx < (int)slot_ids.size()) {
                    entry.slot_ids.push_back(slot_ids[idx]);
                }
            }
        }

        // Tokens
        for (int t = s; t < e; ++t) {
            entry.tokens.push_back(token_ids[t]);
        }

        // Prime node designation
        bool is_prime = false;
        for (int slot : entry.slot_ids) {
            if (semantic_prime_slots.count(slot)) {
                is_prime = true;
                break;
            }
        }
        if (!is_prime) {
            float max_idf = 0.0f;
            for (int t : entry.tokens) {
                auto it = inv_index.idf.find(t);
                float idf = (it != inv_index.idf.end() ? it->second : 1.0f);
                if (idf > max_idf) max_idf = idf;
            }
            if (max_idf >= 3.0f) {
                is_prime = true;
            }
        }
        entry.is_prime = is_prime;

        entries.push_back(std::move(entry));
    }

    // 8a. Assign entity_ids to all entries via token overlap with primes.
    // For each non-prime entry, find the prime whose token set has the
    // highest overlap with this entry's tokens.
    struct PrimeInfo {
        size_t idx;
        std::unordered_set<int32_t> token_set;
        int32_t start_idx;
    };
    std::vector<PrimeInfo> prime_infos;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].is_prime) {
            entries[i].entity_id = entries[i].start_idx;
            PrimeInfo pi;
            pi.idx = i;
            pi.start_idx = entries[i].start_idx;
            pi.token_set.insert(entries[i].tokens.begin(), entries[i].tokens.end());
            prime_infos.push_back(std::move(pi));
        }
    }

    if (!prime_infos.empty()) {
        for (auto& entry : entries) {
            if (entry.is_prime) continue;
            std::unordered_set<int32_t> entry_tokens(entry.tokens.begin(), entry.tokens.end());
            int best_overlap = 0;
            int32_t best_entity = -1;
            for (const auto& pi : prime_infos) {
                int shared = 0;
                for (int32_t t : entry_tokens) {
                    if (pi.token_set.count(t)) ++shared;
                }
                if (shared > best_overlap) {
                    best_overlap = shared;
                    best_entity = pi.start_idx;
                } else if (shared == best_overlap && shared > 0) {
                    if (best_entity == -1 || std::abs(entry.start_idx - pi.start_idx) < std::abs(entry.start_idx - best_entity)) {
                        best_entity = pi.start_idx;
                    }
                }
            }
            if (best_entity == -1) {
                // No token overlap — fall back to positional proximity
                int min_dist = INT_MAX;
                for (const auto& pi : prime_infos) {
                    int dist = std::abs(entry.start_idx - pi.start_idx);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_entity = pi.start_idx;
                    }
                }
            }
            entry.entity_id = best_entity;
        }
    }

    // 8b. Build graph connections with entity-aware dampening.
    // Cross-entity edges receive a 0.3× weight penalty to prevent graph
    // walks from propagating one entity's properties into another's
    // retrieval set.
    const float CROSS_ENTITY_DAMPEN = 0.3f;
    int num_entries = entries.size();
    for (int i = 0; i < num_entries; ++i) {
        auto& entry_i = entries[i];
        std::unordered_set<int> tokens_i(entry_i.tokens.begin(), entry_i.tokens.end());

        for (int j = i + 1; j < num_entries; ++j) {
            auto& entry_j = entries[j];
            std::unordered_set<int> tokens_j(entry_j.tokens.begin(), entry_j.tokens.end());

            // Lexical overlap
            bool lexical_overlap = false;
            for (int t : tokens_i) {
                if (tokens_j.count(t) && !stop_token_ids.count(t)) {
                    lexical_overlap = true;
                    break;
                }
            }

            // Temporal distance
            int temporal_dist = std::abs(entry_i.start_idx - entry_j.start_idx);
            bool is_temporal_adjacent = (temporal_dist < 512);

            // Descriptor similarity
            float sim_val = 0.0f;
            for (int r = 0; r < desc_dim; ++r) {
                sim_val += entry_i.descriptor[r] * entry_j.descriptor[r];
            }
            bool is_similar = (sim_val >= 0.3f);

            if (lexical_overlap || is_temporal_adjacent || is_similar) {
                float w_lex = lexical_overlap ? 1.0f : 0.0f;
                float w_temp = std::max(0.0f, 1.0f - (temporal_dist / 512.0f));
                float w_sim = std::max(0.0f, sim_val);

                float weight = 0.4f * w_sim + 0.4f * w_lex + 0.2f * w_temp;

                // Entity-aware dampening: penalise cross-entity edges
                int32_t eid_i = entry_i.entity_id;
                int32_t eid_j = entry_j.entity_id;
                if (eid_i != -1 && eid_j != -1 && eid_i != eid_j) {
                    weight *= CROSS_ENTITY_DAMPEN;
                }

                entry_i.neighbors.push_back(j);
                entry_i.weights.push_back(weight);

                entry_j.neighbors.push_back(i);
                entry_j.weights.push_back(weight);
            }
        }
    }
}

static std::vector<FactEntry> merge_adjacent_entries(const std::vector<FactEntry>& entries, int num_layers, int F_test) {
    if (entries.empty()) return {};
    std::vector<FactEntry> sorted_entries = entries;
    std::sort(sorted_entries.begin(), sorted_entries.end(),
              [](const FactEntry& a, const FactEntry& b) { return a.start_idx < b.start_idx; });
    std::vector<FactEntry> merged;
    FactEntry curr = sorted_entries[0];
    for (size_t i = 1; i < sorted_entries.size(); ++i) {
        const auto& next_entry = sorted_entries[i];
        if (next_entry.start_idx == curr.end_idx) {
            int curr_len = curr.end_idx - curr.start_idx;
            int next_len = next_entry.end_idx - next_entry.start_idx;
            int new_len = curr_len + next_len;
            
            std::vector<float> new_K(num_layers * F_test * new_len);
            std::vector<float> new_V(num_layers * F_test * new_len);
            
            for (int l = 0; l < num_layers; ++l) {
                std::memcpy(new_K.data() + l * F_test * new_len,
                            curr.K.data() + l * F_test * curr_len,
                            curr_len * F_test * sizeof(float));
                std::memcpy(new_V.data() + l * F_test * new_len,
                            curr.V.data() + l * F_test * curr_len,
                            curr_len * F_test * sizeof(float));
                            
                std::memcpy(new_K.data() + l * F_test * new_len + curr_len * F_test,
                            next_entry.K.data() + l * F_test * next_len,
                            next_len * F_test * sizeof(float));
                std::memcpy(new_V.data() + l * F_test * new_len + curr_len * F_test,
                            next_entry.V.data() + l * F_test * next_len,
                            next_len * F_test * sizeof(float));
            }
            
            curr.end_idx = next_entry.end_idx;
            curr.K = std::move(new_K);
            curr.V = std::move(new_V);
            
            std::unordered_set<int32_t> slots(curr.slot_ids.begin(), curr.slot_ids.end());
            slots.insert(next_entry.slot_ids.begin(), next_entry.slot_ids.end());
            curr.slot_ids.assign(slots.begin(), slots.end());
            
            curr.tokens.insert(curr.tokens.end(), next_entry.tokens.begin(), next_entry.tokens.end());
            curr.current_sim = std::max(curr.current_sim, next_entry.current_sim);
        } else {
            merged.push_back(curr);
            curr = next_entry;
        }
    }
    merged.push_back(curr);
    return merged;
}

std::vector<FactEntry> FactualExactStore::query(
    const float* Q,
    int H_q,
    int head_dim,
    const float* W_proj,
    int desc_dim,
    float threshold,
    const std::unordered_set<int32_t>* active_slots
) const {
    if (entries.empty() || !W_proj) return {};

    // 1. Mean query: [head_dim]
    std::vector<float> avg_q(head_dim, 0.0f);
    for (int h = 0; h < H_q; ++h) {
        const float* qh = Q + h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            avg_q[d] += qh[d];
        }
    }
    for (int d = 0; d < head_dim; ++d) {
        avg_q[d] /= H_q;
    }

    // 2. Project query descriptor: [desc_dim]
    std::vector<float> q_desc(desc_dim, 0.0f);
    float norm_sq = 0.0f;
    for (int r = 0; r < desc_dim; ++r) {
        float val = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            val += avg_q[d] * W_proj[r * head_dim + d];
        }
        q_desc[r] = val;
        norm_sq += val * val;
    }
    float norm = std::sqrt(norm_sq) + 1e-8f;
    for (int r = 0; r < desc_dim; ++r) {
        q_desc[r] /= norm;
    }

    // 3. Base Layer: Candidate indexes matching active_slots
    std::unordered_set<int> candidate_indices;
    if (active_slots) {
        for (size_t idx = 0; idx < entries.size(); ++idx) {
            const auto& entry = entries[idx];
            bool has_slot = false;
            for (int slot : entry.slot_ids) {
                if (active_slots->count(slot)) {
                    has_slot = true;
                    break;
                }
            }
            if (has_slot) {
                candidate_indices.insert(idx);
            }
        }
    } else {
        for (size_t idx = 0; idx < entries.size(); ++idx) {
            candidate_indices.insert(idx);
        }
    }

    // 4. Factual Prime Node Activation (seeds)
    std::vector<std::pair<int, float>> prime_seeds;
    for (size_t idx = 0; idx < entries.size(); ++idx) {
        const auto& entry = entries[idx];
        if (entry.is_prime) {
            float sim = 0.0f;
            for (int r = 0; r < desc_dim; ++r) {
                sim += q_desc[r] * entry.descriptor[r];
            }
            if (sim >= threshold) {
                prime_seeds.push_back({(int)idx, sim});
            }
        }
    }

    // 5. Factual Graph Walk
    std::unordered_map<int, float> walk_candidates;
    for (const auto& seed : prime_seeds) {
        int seed_idx = seed.first;
        float seed_sim = seed.second;
        walk_candidates[seed_idx] = seed_sim;
        const auto& entry = entries[seed_idx];
        for (size_t nb_i = 0; nb_i < entry.neighbors.size(); ++nb_i) {
            int nb_idx = entry.neighbors[nb_i];
            float weight = entry.weights[nb_i];
            float prop_sim = seed_sim * weight;
            auto it = walk_candidates.find(nb_idx);
            if (it == walk_candidates.end() || prop_sim > it->second) {
                walk_candidates[nb_idx] = prop_sim;
            }
        }
    }

    // 6. Merge candidates
    std::vector<std::pair<FactEntry, float>> merged_results;
    std::unordered_set<int> all_candidate_idxs = candidate_indices;
    for (const auto& pair : walk_candidates) {
        all_candidate_idxs.insert(pair.first);
    }

    for (int idx : all_candidate_idxs) {
        const auto& entry = entries[idx];
        float sim = 0.0f;
        for (int r = 0; r < desc_dim; ++r) {
            sim += q_desc[r] * entry.descriptor[r];
        }

        bool passes_main = (sim >= threshold);
        bool passes_relaxed = false;
        if (active_slots) {
            bool has_slot = false;
            for (int slot : entry.slot_ids) {
                if (active_slots->count(slot)) {
                    has_slot = true;
                    break;
                }
            }
            if (has_slot && sim >= 0.15f) {
                passes_relaxed = true;
            }
        }
        auto walk_it = walk_candidates.find(idx);
        // Two-condition walk gate (mirrors Python logic):
        //   1. walk_score >= 0.20 (propagated relevance from seed)
        //   2. sim >= 0.10 (some direct distributional relevance)
        //      OR walk_score >= 0.30 (strong edge — trust graph topology)
        // Prevents zero-sim entries via weak temporal edges from being returned.
        const float WALK_THRESHOLD = 0.20f;
        const float WALK_STRONG    = 0.30f;
        const float WALK_MIN_SIM   = 0.10f;
        bool passes_walk = false;
        if (walk_it != walk_candidates.end()) {
            float ws = walk_it->second;
            passes_walk = (ws >= WALK_THRESHOLD && (sim >= WALK_MIN_SIM || ws >= WALK_STRONG));
        }

        if (passes_main || passes_relaxed || passes_walk) {
            float final_score = sim;
            if (walk_it != walk_candidates.end() && walk_it->second > final_score) {
                final_score = walk_it->second;
            }
            entry.current_sim = final_score;
            merged_results.push_back({entry, final_score});
        }
    }

    // Fallback
    if (active_slots && merged_results.empty()) {
        std::vector<std::pair<FactEntry, float>> fallback_matches;
        for (int idx : candidate_indices) {
            const auto& entry = entries[idx];
            float sim = 0.0f;
            for (int r = 0; r < desc_dim; ++r) {
                sim += q_desc[r] * entry.descriptor[r];
            }
            if (sim >= 0.15f) {
                entry.current_sim = sim;
                fallback_matches.push_back({entry, sim});
            }
        }
        std::sort(fallback_matches.begin(), fallback_matches.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        if (!fallback_matches.empty()) {
            FactEntry fb_entry = fallback_matches[0].first;
            fb_entry.current_sim = fallback_matches[0].second;
            merged_results.push_back({fb_entry, fallback_matches[0].second});
        }
    }

    std::sort(merged_results.begin(), merged_results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Top 5 only — with 5% selection, each entry is a tight, high-precision span.
    // Returning more than 5 at this precision level floods the VSL with too many candidates.
    std::vector<FactEntry> results;
    int limit = std::min(5, (int)merged_results.size());
    for (int i = 0; i < limit; ++i) {
        FactEntry top_entry = merged_results[i].first;
        top_entry.current_sim = merged_results[i].second;
        results.push_back(top_entry);
    }
    return merge_adjacent_entries(results, num_layers, F_test);
}

} // namespace diffkv
