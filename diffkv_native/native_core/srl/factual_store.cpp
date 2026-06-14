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
    clear();
    int L = token_ids.size();
    if (L == 0 || k_activations.empty()) return;
    int num_layers = k_activations.size();
    int F_test = kv_heads * head_dim;
    int max_l = k_activations[0].size() / F_test;
    if (L > max_l) {
        L = max_l;
    }
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

        // 5d. Mandatory prime slot coverage — each cluster center (semantic
        // prime slot) must have at least one token in the factual mask even
        // if all its tokens fall below the global 5% salience threshold.
        // Without this, blocks at the structural center of a cluster are
        // silently excluded and their concepts (e.g. "Riemann sheets" in the
        // EP cluster) cannot be associated with the correct entity, causing
        // topology leakage into neighbouring clusters.
        if (block_size > 0 && !slot_ids.empty() && !semantic_prime_slots.empty()) {
            for (int32_t slot : semantic_prime_slots) {
                auto it = std::find(slot_ids.begin(), slot_ids.end(), slot);
                if (it == slot_ids.end()) continue;
                int block_idx = (int)(it - slot_ids.begin());
                int tok_start = block_idx * block_size;
                int tok_end = std::min(tok_start + block_size, L);
                if (tok_start >= L) continue;
                bool any_covered = false;
                for (int t = tok_start; t < tok_end; ++t) {
                    if (factual_mask[t]) { any_covered = true; break; }
                }
                if (!any_covered) {
                    // Force the highest-salience token in this slot and
                    // expand by the context window.
                    int peak = tok_start;
                    float peak_sal = total_salience[tok_start];
                    for (int t = tok_start + 1; t < tok_end; ++t) {
                        if (total_salience[t] > peak_sal) {
                            peak_sal = total_salience[t];
                            peak = t;
                        }
                    }
                    constexpr int FORCED_WIN = 3;  // mirrors CONTEXT_WINDOW in 5b
                    int lo = std::max(0, peak - FORCED_WIN);
                    int hi = std::min(L, peak + FORCED_WIN + 1);
                    for (int t = lo; t < hi; ++t) factual_mask[t] = true;
                }
            }
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

        // 3-token source prefix for quote-grounded connective gating (RC2)
        {
            int prefix_start = std::max(0, s - 3);
            for (int t = prefix_start; t < s; ++t) {
                entry.prefix_tokens.push_back(token_ids[t]);
            }
        }

        // Prime node designation — RC7: require BOTH rarity (IDF) AND lookback
        // reference (Eagle-R) so that jargon-only spans don't become phantom entities.
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
            // Eagle-R: max lookback score for this span
            bool have_eagle = !eagle_scores.empty();
            float max_r = 0.0f;
            if (have_eagle) {
                for (int t = s; t < e && t < (int)eagle_scores.size(); ++t) {
                    if (eagle_scores[t] > max_r) max_r = eagle_scores[t];
                }
            }
            if (have_eagle) {
                // RC7: an entity is a rare term the document refers back to —
                // require BOTH rarity AND reference so one-off jargon doesn't
                // spawn a phantom entity.
                if (max_idf >= 3.0f && max_r >= 0.3f) {
                    is_prime = true;
                } else if (max_r >= 1.5f) {
                    // Highly referenced regardless of rarity → entity prime
                    is_prime = true;
                }
            } else {
                // RC7 fallback: no reference signal was computed — cannot demand a
                // reference we never measured, so fall back to rarity-only.
                if (max_idf >= 3.0f) {
                    is_prime = true;
                }
            }
        }
        entry.is_prime = is_prime;

        // For primes, record the single most distinguishing (highest-IDF) token,
        // used by assign_entities() to bind properties that explicitly name their
        // entity, overriding misleading low-IDF token overlap (RC4).
        if (is_prime) {
            int32_t best_tok = -1;
            float best_idf = 0.0f;
            for (int32_t t : entry.tokens) {
                auto it = inv_index.idf.find(t);
                float idf = (it != inv_index.idf.end() ? it->second : 1.0f);
                if (idf >= 3.0f && idf > best_idf) {
                    best_idf = idf;
                    best_tok = t;
                }
            }
            entry.distinguishing_token = best_tok;
        }

        entries.push_back(std::move(entry));
    }

    // 8a. Assign entity_ids: bind each property span to the prime that owns it.
    //
    // Plain token-overlap mis-binds related concepts: a property of EP2 such as
    // "codimension 2 while" shares the filler "while" with the EP3 span and names
    // neither entity, so it gets pulled to EP3 (RC4).  Absolute positional
    // proximity is also ambiguous in interleaved comparison text.  Ownership is
    // decided in priority order (mirrors Python FactualExactStore._assign_entities):
    //   1. distinguishing-token override — span explicitly names exactly one prime
    //   2. IDF-weighted overlap — span names several primes
    //   3. nearest-PRECEDING prime in document order (reading-order ownership)
    struct PrimeInfo {
        std::unordered_set<int32_t> token_set;
        int32_t start_idx;
        int32_t distinguishing_token;
    };
    std::vector<PrimeInfo> prime_infos;
    for (auto& fe : entries) {
        if (fe.is_prime) {
            fe.entity_id = fe.start_idx;
            PrimeInfo pi;
            pi.start_idx = fe.start_idx;
            pi.distinguishing_token = fe.distinguishing_token;
            pi.token_set.insert(fe.tokens.begin(), fe.tokens.end());
            prime_infos.push_back(std::move(pi));
        }
    }

    if (!prime_infos.empty()) {
        // Sorted prime positions for reading-order (nearest-preceding) ownership.
        std::vector<int32_t> prime_starts;
        prime_starts.reserve(prime_infos.size());
        for (const auto& pi : prime_infos) prime_starts.push_back(pi.start_idx);
        std::sort(prime_starts.begin(), prime_starts.end());

        auto tok_idf = [&](int32_t t) -> float {
            auto it = inv_index.idf.find(t);
            return (it != inv_index.idf.end() ? it->second : 1.0f);
        };

        for (auto& entry : entries) {
            if (entry.is_prime) continue;
            std::unordered_set<int32_t> entry_tokens(entry.tokens.begin(), entry.tokens.end());

            // 1./2. Distinguishing-token signal.
            std::vector<int32_t> named_primes;
            for (const auto& pi : prime_infos) {
                if (pi.distinguishing_token >= 0 && entry_tokens.count(pi.distinguishing_token)) {
                    named_primes.push_back(pi.start_idx);
                }
            }
            if (named_primes.size() == 1) {
                entry.entity_id = named_primes[0];
                continue;
            }
            if (named_primes.size() > 1) {
                std::unordered_set<int32_t> named_set(named_primes.begin(), named_primes.end());
                int32_t best_entity = -1;
                float best_score = -1.0f;
                for (const auto& pi : prime_infos) {
                    if (!named_set.count(pi.start_idx)) continue;
                    float score = 0.0f;
                    for (int32_t t : entry_tokens) {
                        if (pi.token_set.count(t)) score += tok_idf(t);
                    }
                    if (score > best_score) {
                        best_score = score;
                        best_entity = pi.start_idx;
                    }
                }
                entry.entity_id = best_entity;
                continue;
            }

            // 3. Reading-order ownership: nearest prime introduced before this span.
            int32_t owner = -1;
            for (int32_t ps : prime_starts) {
                if (ps <= entry.start_idx) owner = ps; else break;
            }
            if (owner == -1) {
                // No prime precedes — fall back to nearest prime overall.
                int min_dist = INT_MAX;
                for (int32_t ps : prime_starts) {
                    int dist = std::abs(entry.start_idx - ps);
                    if (dist < min_dist) { min_dist = dist; owner = ps; }
                }
            }
            entry.entity_id = owner;
        }
    }

    // 8a-rc3. Stamp each entry with its owning entity's signature so an
    // entity-biased query can rank the queried entity's spans above
    // shared-vocabulary spans of other entities (RC3).
    entity_distinguishing.clear();
    for (const auto& fe : entries) {
        if (fe.is_prime && fe.distinguishing_token >= 0) {
            entity_distinguishing[fe.start_idx] = fe.distinguishing_token;
        }
    }
    for (auto& entry : entries) {
        auto it = entity_distinguishing.find(entry.entity_id);
        if (it != entity_distinguishing.end()) {
            entry.entity_sig = entity_signature(it->second);
        }
    }

    // 8b-rc1. Extract typed directed triples: (prime → bridge → value).
    // For each prime entry P, scan forward for property spans owned by the same
    // entity.  The document tokens between P.end_idx and V.start_idx form the
    // "bridge" (relational connectives: "is", "has", "exhibits", etc.).
    // Stored as triple_sequences on the prime for injection at decode time.
    {
        // Build set of relational token IDs (low-IDF binders)
        std::unordered_set<int32_t> rel_tid_set;
        for (int i_t = 0; i_t < L; ++i_t) {
            int32_t tid = token_ids[i_t];
            auto it = inv_index.idf.find(tid);
            float idf = (it != inv_index.idf.end()) ? it->second : 2.0f;
            if (idf < 1.5f) rel_tid_set.insert(tid);
        }

        for (auto& p_entry : entries) {
            if (!p_entry.is_prime) continue;
            int32_t p_entity = p_entry.start_idx;
            int p_end = p_entry.end_idx;

            for (const auto& v_entry : entries) {
                if (v_entry.is_prime) continue;
                if (v_entry.entity_id != p_entity) continue;
                if (v_entry.start_idx < p_end) continue;
                int gap = v_entry.start_idx - p_end;
                if (gap > 96) continue;

                // Build bridge tokens
                std::vector<int32_t> bridge;
                for (int t = p_end; t < v_entry.start_idx; ++t) {
                    bridge.push_back(token_ids[t]);
                }
                if (bridge.size() > 48) continue; // raised from 8: with 5%-sparse
                // span coverage bridges between adjacent spans are consistently
                // 20-30 tokens long, silently excluding every valid triple

                // Require at least one relational token in bridge
                bool has_rel = false;
                for (int32_t tid : bridge) {
                    if (rel_tid_set.count(tid)) { has_rel = true; break; }
                }
                if (!has_rel) continue;

                // Build triple sequence: bridge + value tokens
                std::vector<int32_t> triple_seq = bridge;
                triple_seq.insert(triple_seq.end(),
                                  v_entry.tokens.begin(), v_entry.tokens.end());

                // Deduplicate
                bool already_present = false;
                for (const auto& ts : p_entry.triple_sequences) {
                    if (ts == triple_seq) { already_present = true; break; }
                }
                if (!already_present && !triple_seq.empty()) {
                    p_entry.triple_sequences.push_back(std::move(triple_seq));
                }
            }
        }
    }

    // 8c-dx1. DX1: Tag definition spans.
    // A non-prime span whose owning prime is ≤ 32 tokens away via a short copular
    // bridge (IDF < 1.0) is marked is_definition = true.  query() boosts these
    // entries by 1.5× so definitions always survive the coherence cap.
    {
        // Build prime_by_entity map
        std::unordered_map<int32_t, FactEntry*> prime_by_entity;
        for (auto& fe : entries) {
            if (fe.is_prime) prime_by_entity[fe.start_idx] = &fe;
        }
        for (auto& entry : entries) {
            if (entry.is_prime || entry.entity_id == -1) continue;
            auto it = prime_by_entity.find(entry.entity_id);
            if (it == prime_by_entity.end()) continue;
            const FactEntry* p_entry = it->second;
            if (entry.start_idx < p_entry->end_idx) continue;
            int gap = entry.start_idx - p_entry->end_idx;
            if (gap > 32) continue;
            int bridge_len = gap;
            if (bridge_len > 5) continue;
            // Definitional bridge contains a copular verb with IDF < 1.0
            bool has_copula = false;
            for (int t = p_entry->end_idx; t < entry.start_idx && t < (int)token_ids.size(); ++t) {
                int32_t tid = token_ids[t];
                float idf = inv_index.idf.count(tid) ? inv_index.idf.at(tid) : 2.0f;
                if (idf < 1.0f) { has_copula = true; break; }
            }
            if (has_copula || bridge_len <= 3) {
                entry.is_definition = true;
            }
        }
    }

    // 8e. Build graph connections with entity-aware dampening.
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
        if (next_entry.start_idx == curr.end_idx && curr.entity_id == next_entry.entity_id) {
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
    const std::unordered_set<int32_t>* active_slots,
    const std::unordered_set<int32_t>* query_entity_bias
) const {
    if (entries.empty() || !W_proj) return {};

    // RC3 — build the entity-bias signature from the queried entities'
    // distinguishing tokens.  Empty when no bias supplied (null-safe: ranking is
    // then identical to before).
    const float ENTITY_BIAS_WEIGHT = 0.15f;
    std::vector<float> q_sig;
    if (query_entity_bias && !query_entity_bias->empty()) {
        std::vector<float> acc(ENTITY_SIG_DIM, 0.0f);
        double nrm = 0.0;
        for (int32_t eid : *query_entity_bias) {
            auto it = entity_distinguishing.find(eid);
            if (it == entity_distinguishing.end()) continue;
            std::vector<float> s = entity_signature(it->second);
            for (int i = 0; i < ENTITY_SIG_DIM; ++i) acc[i] += s[i];
        }
        for (int i = 0; i < ENTITY_SIG_DIM; ++i) nrm += (double)acc[i] * acc[i];
        nrm = std::sqrt(nrm);
        if (nrm > 1e-8) {
            for (int i = 0; i < ENTITY_SIG_DIM; ++i) acc[i] = (float)(acc[i] / nrm);
            q_sig = std::move(acc);
        }
    }

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
            // RC3: nudge ranking toward the queried entity's spans (ordering only;
            // the pass/fail gates above are untouched so recall is unchanged).
            if (!q_sig.empty() && !entry.entity_sig.empty()) {
                float dp = 0.0f;
                for (int i = 0; i < ENTITY_SIG_DIM && i < (int)entry.entity_sig.size(); ++i) {
                    dp += q_sig[i] * entry.entity_sig[i];
                }
                final_score += ENTITY_BIAS_WEIGHT * dp;
            }
            // DX1: definition spans get boosted retrieval priority so they
            // always sort above tangential context in the coherence cap.
            if (entry.is_definition) {
                final_score = std::min(final_score * 1.5f, 1.0f);
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

    // RC6 — entity-proportional budget: scale K with matched prime count so each
    // entity gets coverage in comparison answers (fixed top-5 drops one entity).
    int n_matched_primes = static_cast<int>(prime_seeds.size());
    int k_budget = std::max(5, n_matched_primes * 3 + 2);
    std::vector<FactEntry> results;
    int limit = std::min(k_budget, (int)merged_results.size());
    for (int i = 0; i < limit; ++i) {
        FactEntry top_entry = merged_results[i].first;
        top_entry.current_sim = merged_results[i].second;
        results.push_back(top_entry);
    }
    return merge_adjacent_entries(results, num_layers, F_test);
}

} // namespace diffkv
