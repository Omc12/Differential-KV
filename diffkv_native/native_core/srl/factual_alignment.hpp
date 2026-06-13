// factual_alignment.hpp
#pragma once

#include "native_core/srl/session_srl_state.hpp"
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <cmath>

namespace diffkv {

inline std::string clean_token_text(const std::string& text) {
    std::string cleaned = "";
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cleaned += std::tolower(static_cast<unsigned char>(c));
        }
    }
    return cleaned;
}

template <typename ModelType>
inline const std::unordered_set<int32_t>& get_helper_token_ids_cpp(ModelType& model) {
    static std::unordered_set<int32_t> cached_helper_ids;
    static bool is_initialized = false;
    if (is_initialized) {
        return cached_helper_ids;
    }

    static const std::unordered_set<std::string> ALLOWED_HELPER_WORDS = {
        "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "that", "this", "these", "those", "each", "every", "some", "any", "no", "all", "both",
        "either", "neither", "another", "other", "such", "what", "a", "an", "the",
        "of", "in", "to", "for", "with", "on", "at", "by", "from", "about", "as", "into", "through", "during", "before", "after",
        "above", "below", "up", "down", "over", "under", "between", "among", "out", "off", "within", "without", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "and", "or", "but", "so", "yet", "nor", "although",
        "because", "since", "unless", "until", "while", "whereas", "if", "else", "than",
        "is", "was", "were", "are", "be", "been", "being", "am", "have", "has", "had", "having", "do", "does", "did", "doing",
        "correct", "correctly", "faithful", "faithfully", "verbatim", "missing", "present", "clear", "clearly",
        "uncertain", "certain", "provide", "provided", "provides", "note", "noted", "notes",
        "not", "only", "very", "too", "just", "well", "also", "now", "first", "second", "third", "one", "two", "three",
        "case", "cases", "correspond", "corresponds", "corresponding", "example", "examples", "result", "results",
        "value", "values", "number", "numbers", "term", "terms", "word", "words", "mean", "means", "meant", "meaning",
        "define", "defines", "defined", "definition", "definitions", "represent", "represents", "represented",
        "explanation", "explanations", "statement", "statements", "use", "uses", "used", "using", "make", "makes",
        "made", "making", "take", "takes", "taken", "taking", "part", "parts", "point", "points", "set", "sets"
    };

    int n_vocab = model.get_config().n_vocab;
    for (int32_t tok_id = 0; tok_id < n_vocab; ++tok_id) {
        std::string piece = model.token_to_piece(tok_id);
        std::string cleaned = clean_token_text(piece);
        if (cleaned.empty()) {
            cached_helper_ids.insert(tok_id);
        } else if (ALLOWED_HELPER_WORDS.count(cleaned) > 0) {
            cached_helper_ids.insert(tok_id);
        }
    }

    is_initialized = true;
    return cached_helper_ids;
}

// Returns the set of allowed token IDs for the current decode step.
//
// LOCK ACTIVE: allowed = helper_ids ∪ {suffix[0] for each active suffix}
//
// NO ACTIVE LOCK (entity-filtered):
//   allowed = helper_ids ∪ {seq[0] for same-entity and prime sequences}
//   - current_entity_id == -1: all sequence starts allowed (no context yet).
//   - Otherwise: only sequences whose entity_id matches current_entity_id, or
//     sequences with unknown entity (-1), or prime sequences (to allow entity
//     transitions, e.g. generating "EP3" after finishing the EP2 section).
// ── Contrastive Category Anchor, Coherence Cap, & Entity-Subgraph Tagging ──────
// Process the active factual sequences at the current step:
// 1. Contrastive Category Anchor (CA): if we have generated enough tokens, lock to dominant entity
// 2. Coherence Cap: limit the active sequences to 8, sorted by retrieval similarity
// 3. Entity-Subgraph Tagging: tag sequences with their assigned entity_ids (Jaccard token-overlap based)
inline void process_and_tag_vsl_step(SessionSRLState& srl_state) {
    // 1. Contrastive Category Anchor
    {
        std::unordered_set<int32_t> recent_set;
        int rgt_size = (int)srl_state.recent_generated_tokens.size();
        int rgt_start = std::max(0, rgt_size - 30);
        for (int ri = rgt_start; ri < rgt_size; ++ri) {
            recent_set.insert(srl_state.recent_generated_tokens[ri]);
        }

        int best_prime_pos    = -1;
        int best_overlap      = 0;
        int second_best_overlap = 0;
        int active_prime_count  = 0;

        for (const auto& fe : srl_state.factual_store.entries) {
            if (fe.is_prime) {
                int overlap = 0;
                for (int32_t t : fe.tokens) {
                    if (recent_set.count(t)) overlap++;
                }
                if (overlap >= 1) {
                    active_prime_count++;
                    if (overlap > best_overlap) {
                        second_best_overlap = best_overlap;
                        best_overlap   = overlap;
                        best_prime_pos = fe.start_idx;
                    } else if (overlap > second_best_overlap) {
                        second_best_overlap = overlap;
                    }
                }
            }
        }

        int effective_prime_pos = -1;
        if (active_prime_count == 1 && best_prime_pos >= 0) {
            effective_prime_pos = best_prime_pos;
        } else if (active_prime_count == 2 && best_prime_pos >= 0
                   && best_overlap >= 2
                   && best_overlap >= 2 * (second_best_overlap + 1)) {
            effective_prime_pos = best_prime_pos;
        }

        if (effective_prime_pos >= 0) {
            srl_state.current_entity_id = effective_prime_pos;

            std::vector<std::vector<int32_t>> filtered_seqs;
            std::unordered_set<int32_t> filtered_toks;
            for (const auto& seq : srl_state.current_step_factual_sequences) {
                int seq_pos = -1;
                for (const auto& fe : srl_state.factual_store.entries) {
                    if (fe.tokens == seq) { seq_pos = fe.start_idx; break; }
                }
                bool keep = (seq_pos < 0) || (std::abs(seq_pos - effective_prime_pos) < 512);
                if (keep) {
                    filtered_seqs.push_back(seq);
                    filtered_toks.insert(seq.begin(), seq.end());
                }
            }
            if (!filtered_seqs.empty()) {
                srl_state.current_step_factual_sequences = std::move(filtered_seqs);
                srl_state.current_step_factual_tokens    = std::move(filtered_toks);
            }
        }
    }

    // 2. Coherence Cap
    if (srl_state.current_step_factual_sequences.size() > 8) {
        size_t n_seqs = srl_state.current_step_factual_sequences.size();
        std::vector<float> seq_sims(n_seqs, 0.0f);
        for (size_t si = 0; si < n_seqs; ++si) {
            const auto& seq = srl_state.current_step_factual_sequences[si];
            float best_sim = 0.0f;
            for (const auto& fe : srl_state.factual_store.entries) {
                if (fe.tokens == seq && fe.current_sim > best_sim) {
                    best_sim = fe.current_sim;
                }
            }
            seq_sims[si] = best_sim;
        }
        std::vector<size_t> order(n_seqs);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return seq_sims[a] > seq_sims[b];
        });
        order.resize(8);
        std::vector<std::vector<int32_t>> top_seqs;
        std::unordered_set<int32_t> top_toks;
        for (size_t idx : order) {
            top_seqs.push_back(srl_state.current_step_factual_sequences[idx]);
            top_toks.insert(srl_state.current_step_factual_sequences[idx].begin(), srl_state.current_step_factual_sequences[idx].end());
        }
        if (!top_seqs.empty()) {
            srl_state.current_step_factual_sequences = std::move(top_seqs);
            srl_state.current_step_factual_tokens    = std::move(top_toks);
        }
    }

    // 3. Entity-Subgraph Tagging
    {
        const auto& seqs = srl_state.current_step_factual_sequences;
        std::vector<int32_t> ent_ids(seqs.size(), -1);
        std::vector<bool>    is_prime_flags(seqs.size(), false);
        for (size_t si = 0; si < seqs.size(); ++si) {
            int32_t matched_entity = -1;
            bool    seq_prime = false;
            for (const auto& fe : srl_state.factual_store.entries) {
                if (fe.tokens == seqs[si]) {
                    matched_entity = fe.entity_id;
                    seq_prime = fe.is_prime;
                    break;
                }
            }
            ent_ids[si] = matched_entity;
            is_prime_flags[si] = seq_prime;
        }
        srl_state.current_step_sequence_entity_ids = std::move(ent_ids);
        srl_state.current_step_sequence_is_prime   = std::move(is_prime_flags);
    }
}

// Returns the set of allowed token IDs for the current decode step.
inline std::unordered_set<int32_t> get_allowed_tokens_vsl_cpp(
    const SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& helper_ids
) {
    std::unordered_set<int32_t> allowed = helper_ids;

    bool has_active_lock = false;
    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty()) {
            allowed.insert(suffix[0]);
            has_active_lock = true;
        }
    }

    if (!has_active_lock) {
        int32_t current_entity = srl_state.current_entity_id;
        bool dual_mode = srl_state.dual_entity_mode;
        std::unordered_set<int32_t> dual_ids(srl_state.dual_entity_ids.begin(), srl_state.dual_entity_ids.end());
        const auto& entity_ids    = srl_state.current_step_sequence_entity_ids;
        const auto& is_prime_list = srl_state.current_step_sequence_is_prime;
        const auto& seqs          = srl_state.current_step_factual_sequences;

        // Build a mapping of prime tokens by entity
        std::unordered_map<int32_t, std::unordered_set<int32_t>> prime_tokens_by_entity;
        for (size_t i = 0; i < seqs.size(); ++i) {
            bool seq_is_prime = (i < is_prime_list.size()) ? is_prime_list[i] : false;
            int32_t seq_entity = (i < entity_ids.size()) ? entity_ids[i] : -1;
            if (seq_is_prime && seq_entity != -1) {
                prime_tokens_by_entity[seq_entity].insert(seqs[i].begin(), seqs[i].end());
            }
        }

        for (size_t i = 0; i < seqs.size(); ++i) {
            if (seqs[i].empty()) continue;
            int32_t seq_entity  = (i < entity_ids.size())    ? entity_ids[i]    : -1;
            bool    seq_is_prime = (i < is_prime_list.size()) ? is_prime_list[i] : false;

            // Prime sequences are always allowed (they gate entity transitions)
            if (seq_is_prime) {
                allowed.insert(seqs[i][0]);
                continue;
            }

            if (current_entity == -1 && !dual_mode) {
                // No entity context yet — allow all
                allowed.insert(seqs[i][0]);
            } else if (dual_mode) {
                // Dual-entity mode: allow both identified entities, block others
                if (dual_ids.count(seq_entity) || seq_entity == -1) {
                    allowed.insert(seqs[i][0]);
                }
            } else {
                // Single-entity mode: strict entity gating
                if (seq_entity == current_entity) {
                    allowed.insert(seqs[i][0]);
                } else if (seq_entity == -1) {
                    // Unknown-entity restriction: only allow if tokens overlap
                    // with the current entity's prime token set
                    auto it = prime_tokens_by_entity.find(current_entity);
                    if (it != prime_tokens_by_entity.end()) {
                        const auto& prime_toks = it->second;
                        bool has_overlap = false;
                        for (int32_t t : seqs[i]) {
                            if (prime_toks.count(t)) {
                                has_overlap = true;
                                break;
                            }
                        }
                        if (has_overlap) {
                            allowed.insert(seqs[i][0]);
                        }
                    } else {
                        // No prime tokens known for this entity — allow as fallback
                        allowed.insert(seqs[i][0]);
                    }
                }
            }
        }
    }

    return allowed;
}

// Advance the VSL lock state after a token is generated.
// Helpers pass through; threshold 4→12 so normal bridge phrases don't drop the lock.
// When starting a new lock from a sequence-start token, update current_entity_id
// from that sequence's entity tag so subsequent fallback steps are entity-consistent.
inline void update_vsl_state_cpp(
    int32_t token_id,
    SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& helper_ids
) {
    if (helper_ids.count(token_id) > 0) {
        srl_state.vsl_consecutive_helpers++;
        if (srl_state.vsl_consecutive_helpers >= 12) {
            srl_state.vsl_active_candidates.clear();
        }
        return;
    }

    std::vector<std::vector<int32_t>> new_candidates;

    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty() && suffix[0] == token_id) {
            std::vector<int32_t> next_suffix(suffix.begin() + 1, suffix.end());
            new_candidates.push_back(next_suffix);
        }
    }

    bool has_active_lock = false;
    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty()) {
            has_active_lock = true;
            break;
        }
    }

    if (new_candidates.empty() && !has_active_lock) {
        // In entity-filtered fallback mode only sequence-START tokens are reachable,
        // so we only match seq[0] here.  Record the entity of the entered sequence.
        const auto& seqs      = srl_state.current_step_factual_sequences;
        const auto& entity_ids = srl_state.current_step_sequence_entity_ids;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (!seqs[i].empty() && seqs[i][0] == token_id) {
                std::vector<int32_t> next_suffix(seqs[i].begin() + 1, seqs[i].end());
                new_candidates.push_back(next_suffix);
                int32_t seq_entity = (i < entity_ids.size()) ? entity_ids[i] : -1;
                if (seq_entity != -1) {
                    srl_state.current_entity_id = seq_entity;
                }
            }
        }
    }

    srl_state.vsl_active_candidates = new_candidates;
    srl_state.vsl_consecutive_helpers = 0;
}

template <typename ModelType>
inline bool is_token_id_allowed_cpp(
    int32_t token_id,
    const SessionSRLState& srl_state,
    int32_t last_token,
    ModelType& model
) {
    const auto& helper_ids = get_helper_token_ids_cpp(model);
    auto allowed = get_allowed_tokens_vsl_cpp(srl_state, helper_ids);
    return allowed.count(token_id) > 0;
}

} // namespace diffkv
