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

// RC5 — advance a comparison's per-entity block sequence.  Returns the new
// active index; updates `covered` in place.  Mirrors Python
// advance_comparison_entity.  Advances only once the active entity's prime AND
// at least one of its property tokens have appeared in recent output.
inline int advance_comparison_entity(
    const std::vector<int32_t>& comparison_entities,
    int active_idx,
    std::unordered_set<int32_t>& covered,
    const std::unordered_set<int32_t>& recent_tokens,
    const std::unordered_map<int32_t, std::unordered_set<int32_t>>& prime_tokens_by_entity,
    const std::unordered_map<int32_t, std::unordered_set<int32_t>>& prop_tokens_by_entity
) {
    if (comparison_entities.empty()) return active_idx;
    if (active_idx < 0) active_idx = 0;
    if (active_idx >= (int)comparison_entities.size()) active_idx = (int)comparison_entities.size() - 1;
    int32_t active_eid = comparison_entities[active_idx];

    auto seen_in = [&](const std::unordered_map<int32_t, std::unordered_set<int32_t>>& m) -> bool {
        auto it = m.find(active_eid);
        if (it == m.end()) return false;
        for (int32_t t : it->second) if (recent_tokens.count(t)) return true;
        return false;
    };

    if (seen_in(prime_tokens_by_entity) && seen_in(prop_tokens_by_entity)) {
        covered.insert(active_eid);
        for (int nxt = active_idx + 1; nxt < (int)comparison_entities.size(); ++nxt) {
            if (covered.count(comparison_entities[nxt]) == 0) return nxt;
        }
    }
    return active_idx;
}

// RC8 — split the per-step factual sequences into tokens the current entity may
// emit (licensed) and tokens that belong exclusively to other entities
// (foreign).  Mirrors Python compute_entity_token_license.
inline void compute_entity_token_license(
    const std::vector<std::vector<int32_t>>& sequences,
    const std::vector<int32_t>& entity_ids,
    const std::vector<bool>& is_prime_flags,
    int32_t current_entity,
    std::unordered_set<int32_t>& licensed_out,
    std::unordered_set<int32_t>& foreign_out
) {
    licensed_out.clear();
    foreign_out.clear();
    std::unordered_set<int32_t> other;
    for (size_t i = 0; i < sequences.size(); ++i) {
        if (sequences[i].empty()) continue;
        int32_t eid = (i < entity_ids.size()) ? entity_ids[i] : -1;
        bool isp = (i < is_prime_flags.size()) ? is_prime_flags[i] : false;
        if (isp || eid == -1 || eid == current_entity) {
            licensed_out.insert(sequences[i].begin(), sequences[i].end());
        } else {
            other.insert(sequences[i].begin(), sequences[i].end());
        }
    }
    for (int32_t t : other) {
        if (licensed_out.count(t) == 0) foreign_out.insert(t);
    }
}

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

// RC2: structural helpers = full helpers minus relational binding words.
// Used under SFA+lock to prevent "is", "has", "whereas", "because" etc. from
// bridging two unrelated entity spans within a single locked sequence.
template <typename ModelType>
inline const std::unordered_set<int32_t>& get_structural_helper_token_ids_cpp(ModelType& model) {
    static std::unordered_set<int32_t> cached_structural_ids;
    static bool is_initialized = false;
    if (is_initialized) return cached_structural_ids;

    static const std::unordered_set<std::string> RELATIONAL_BINDING_WORDS = {
        "is", "are", "was", "were", "has", "have", "had",
        "exhibits", "possesses", "contains", "involves", "requires", "lacks", "features",
        "whereas", "while", "although", "but", "however", "yet", "though",
        "notwithstanding", "nevertheless", "nonetheless", "conversely",
        "unlike", "contrast", "instead", "rather",
        "because", "since", "therefore", "hence", "thus", "consequently", "accordingly",
        "than",
    };

    const auto& all_helpers = get_helper_token_ids_cpp(model);
    for (int32_t tok_id : all_helpers) {
        std::string piece = model.token_to_piece(tok_id);
        std::string cleaned;
        for (char c : piece)
            if (std::isalnum(static_cast<unsigned char>(c)))
                cleaned += std::tolower(static_cast<unsigned char>(c));
        if (RELATIONAL_BINDING_WORDS.count(cleaned) == 0) {
            cached_structural_ids.insert(tok_id);
        }
    }

    is_initialized = true;
    return cached_structural_ids;
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
    // 0. Build cache/lookup structures lazily on the first decode step after store build/reset
    if (!srl_state.entries_map_built) {
        srl_state.entries_by_tokens_map.clear();
        srl_state.prime_entries.clear();
        srl_state.cached_triple_hash_to_entity.clear();
        srl_state.cached_prime_tokens_by_entity.clear();
        srl_state.cached_property_tokens_by_entity.clear();
        srl_state.entries_by_token_id.clear();

        auto vec_hash = [](const std::vector<int32_t>& v) -> size_t {
            size_t seed = v.size();
            for (auto x : v) seed ^= (size_t)x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        };

        for (const auto& fe : srl_state.factual_store.entries) {
            srl_state.entries_by_tokens_map[fe.tokens] = &fe;
            for (int32_t tok : fe.tokens) {
                srl_state.entries_by_token_id[tok].push_back(&fe);
            }
            if (fe.is_prime) {
                srl_state.prime_entries.push_back(&fe);
                for (const auto& ts : fe.triple_sequences) {
                    srl_state.cached_triple_hash_to_entity[vec_hash(ts)] = fe.start_idx;
                }
            }
            if (fe.entity_id != -1) {
                if (fe.is_prime) {
                    srl_state.cached_prime_tokens_by_entity[fe.entity_id].insert(fe.tokens.begin(), fe.tokens.end());
                } else {
                    srl_state.cached_property_tokens_by_entity[fe.entity_id].insert(fe.tokens.begin(), fe.tokens.end());
                }
            }
        }
        srl_state.entries_map_built = true;
    }

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

        for (const FactEntry* fe : srl_state.prime_entries) {
            int overlap = 0;
            for (int32_t t : fe->tokens) {
                if (recent_set.count(t)) overlap++;
            }
            if (overlap >= 1) {
                active_prime_count++;
                if (overlap > best_overlap) {
                    second_best_overlap = best_overlap;
                    best_overlap   = overlap;
                    best_prime_pos = fe->start_idx;
                } else if (overlap > second_best_overlap) {
                    second_best_overlap = overlap;
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

            if (!srl_state.dual_entity_mode) {
                std::vector<std::vector<int32_t>> filtered_seqs;
                std::unordered_set<int32_t> filtered_toks;
                for (const auto& seq : srl_state.current_step_factual_sequences) {
                    int seq_pos = -1;
                    auto it = srl_state.entries_by_tokens_map.find(seq);
                    if (it != srl_state.entries_by_tokens_map.end()) {
                        seq_pos = it->second->start_idx;
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
    }

    // 1b. RC5 Comparison Sequencing — in comparison mode the anchor's job is
    // segmentation, not winner-selection: lock to one entity's block and advance
    // only once it is substantively covered.  Overrides the anchor's guess.
    if (srl_state.dual_entity_mode && !srl_state.comparison_entities.empty()) {
        std::unordered_set<int32_t> recent_set;
        int rgt_size = (int)srl_state.recent_generated_tokens.size();
        for (int ri = std::max(0, rgt_size - 30); ri < rgt_size; ++ri) {
            recent_set.insert(srl_state.recent_generated_tokens[ri]);
        }
        int new_idx = advance_comparison_entity(
            srl_state.comparison_entities,
            srl_state.comparison_active_idx,
            srl_state.comparison_covered,
            recent_set,
            srl_state.cached_prime_tokens_by_entity,
            srl_state.cached_property_tokens_by_entity
        );
        srl_state.comparison_active_idx = new_idx;
        srl_state.current_entity_id = srl_state.comparison_entities[new_idx];
    }

    // 2. Coherence Cap — RC6 entity-proportional budget.
    {
        int n_active_primes = 0;
        for (const FactEntry* fe : srl_state.prime_entries) {
            if (fe->current_sim > 0.0f) n_active_primes++;
        }
        // 4 sequences per active entity + 4 overhead; minimum 8.
        int coherence_cap = std::max(8, n_active_primes * 4 + 4);
        if ((int)srl_state.current_step_factual_sequences.size() > coherence_cap) {
            size_t n_seqs = srl_state.current_step_factual_sequences.size();
            std::vector<float> seq_sims(n_seqs, 0.0f);
            for (size_t si = 0; si < n_seqs; ++si) {
                const auto& seq = srl_state.current_step_factual_sequences[si];
                float best_sim = 0.0f;
                auto it = srl_state.entries_by_tokens_map.find(seq);
                if (it != srl_state.entries_by_tokens_map.end()) {
                    best_sim = it->second->current_sim;
                }
                seq_sims[si] = best_sim;
            }
            std::vector<size_t> order(n_seqs);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return seq_sims[a] > seq_sims[b];
            });
            order.resize(static_cast<size_t>(coherence_cap));
            std::vector<std::vector<int32_t>> top_seqs;
            std::unordered_set<int32_t> top_toks;
            for (size_t idx : order) {
                top_seqs.push_back(srl_state.current_step_factual_sequences[idx]);
                for (int32_t t : srl_state.current_step_factual_sequences[idx]) top_toks.insert(t);
            }
            if (!top_seqs.empty()) {
                srl_state.current_step_factual_sequences = std::move(top_seqs);
                srl_state.current_step_factual_tokens    = std::move(top_toks);
            }
        }
    }

    // 3. Entity-Subgraph Tagging — RC4: use stored entry.entity_id (token-overlap
    // matching) not positional proximity.  RC1: triple sequences inherit their
    // prime's entity_id since they were extracted from that prime's context.
    {
        const auto& seqs = srl_state.current_step_factual_sequences;
        std::vector<int32_t> ent_ids(seqs.size(), -1);
        std::vector<bool>    is_prime_flags(seqs.size(), false);
        // RC2: the source tokens preceding each span, used to quote-ground the
        // connectives allowed to bridge into it. empty for triples (their bridge
        // connective is already part of the captured sequence).
        std::vector<std::vector<int32_t>> seq_prefixes(seqs.size());

        auto vec_hash = [](const std::vector<int32_t>& v) -> size_t {
            size_t seed = v.size();
            for (auto x : v) seed ^= (size_t)x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        };

        for (size_t si = 0; si < seqs.size(); ++si) {
            const auto& seq = seqs[si];
            int32_t matched_entity = -1;
            bool    seq_prime = false;
            // Try matching against known entries
            auto it = srl_state.entries_by_tokens_map.find(seq);
            if (it != srl_state.entries_by_tokens_map.end()) {
                const FactEntry* fe = it->second;
                matched_entity = fe->entity_id;
                seq_prime = fe->is_prime;
                seq_prefixes[si] = fe->prefix_tokens;
            }
            // Fallback: check if this is a triple sequence
            if (matched_entity == -1) {
                auto it2 = srl_state.cached_triple_hash_to_entity.find(vec_hash(seq));
                if (it2 != srl_state.cached_triple_hash_to_entity.end()) {
                    matched_entity = it2->second;
                }
            }
            ent_ids[si] = matched_entity;
            is_prime_flags[si] = seq_prime;
        }
        srl_state.current_step_sequence_entity_ids = std::move(ent_ids);
        srl_state.current_step_sequence_is_prime   = std::move(is_prime_flags);
        srl_state.current_step_sequence_prefixes   = std::move(seq_prefixes);
    }
}

// Returns the set of allowed token IDs for the current decode step.
// RC2: When SFA+lock active, relational binding words are stripped from helpers
// so they cannot leak between entity spans.  Pass structural_helper_ids (full
// helpers minus RELATIONAL_BINDING_WORDS) and sfa_active=true to activate this.
inline std::unordered_set<int32_t> get_allowed_tokens_vsl_cpp(
    const SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& helper_ids,
    const std::unordered_set<int32_t>* structural_helper_ids = nullptr,
    bool sfa_active = false
) {
    // Determine whether any lock is currently active.
    bool has_active_lock = false;
    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty()) { has_active_lock = true; break; }
    }

    // RC2: under SFA+lock, switch to structural helpers (no relational binders).
    const std::unordered_set<int32_t>& base_helpers =
        (sfa_active && has_active_lock && structural_helper_ids != nullptr)
        ? *structural_helper_ids
        : helper_ids;

    std::unordered_set<int32_t> allowed = base_helpers;

    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty()) {
            allowed.insert(suffix[0]);
        }
    }

    if (!has_active_lock) {
        int32_t current_entity = srl_state.current_entity_id;
        bool dual_mode = srl_state.dual_entity_mode;
        std::unordered_set<int32_t> dual_ids(srl_state.dual_entity_ids.begin(), srl_state.dual_entity_ids.end());
        const auto& entity_ids    = srl_state.current_step_sequence_entity_ids;
        const auto& is_prime_list = srl_state.current_step_sequence_is_prime;
        const auto& seq_prefixes  = srl_state.current_step_sequence_prefixes;
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

        // Single pass: decide which sequence starts we may enter (entity-filtered)
        // and collect the source-adjacent connectives that bridge into them.
        std::unordered_set<int32_t> enterable_starts;
        std::unordered_set<int32_t> grounded_connectives;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (seqs[i].empty()) continue;
            int32_t seq_entity  = (i < entity_ids.size())    ? entity_ids[i]    : -1;
            bool    seq_is_prime = (i < is_prime_list.size()) ? is_prime_list[i] : false;

            bool enter = false;
            if (seq_is_prime) {
                // Prime sequences are always allowed (they gate entity transitions)
                enter = true;
            } else if (current_entity == -1 && !dual_mode) {
                enter = true;                       // No entity context yet — allow all
            } else if (dual_mode) {
                enter = (dual_ids.count(seq_entity) || seq_entity == -1);
            } else if (seq_entity == current_entity) {
                enter = true;                       // Single-entity strict gating
            } else if (seq_entity == -1) {
                // Unknown-entity restriction: only allow if tokens overlap with the
                // current entity's prime token set (else fallback-allow when unknown).
                auto it = prime_tokens_by_entity.find(current_entity);
                if (it != prime_tokens_by_entity.end()) {
                    const auto& prime_toks = it->second;
                    for (int32_t t : seqs[i]) {
                        if (prime_toks.count(t)) { enter = true; break; }
                    }
                } else {
                    enter = true;
                }
            }

            if (enter) {
                allowed.insert(seqs[i][0]);
                enterable_starts.insert(seqs[i][0]);
                // The tokens that preceded this span in the source are the only
                // connectives allowed to bridge into it (RC2 quote-grounding).
                if (i < seq_prefixes.size()) {
                    for (int32_t pt : seq_prefixes[i]) grounded_connectives.insert(pt);
                }
            }
        }

        // RC2 — Quote-grounded connective gate.  Relational binding words
        // ("is", "has", "because", "whereas", "while", …) are demoted from the
        // free-helper set under SFA and re-admitted ONLY where the source grounds
        // them: they begin a captured sequence (a triple bridge) or were
        // source-adjacent to a span we may now enter.  This stops the model from
        // inventing its own connective scaffold around correct content.
        if (sfa_active && structural_helper_ids != nullptr) {
            std::vector<int32_t> to_remove;
            for (int32_t rid : helper_ids) {
                if (structural_helper_ids->count(rid) == 0
                    && enterable_starts.count(rid) == 0
                    && grounded_connectives.count(rid) == 0) {
                    to_remove.push_back(rid);
                }
            }
            for (int32_t rid : to_remove) allowed.erase(rid);
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
    // RECONSTRUCTION FIX (F24): mirror ACTIVE_RUNTIME update_vsl_state ordering
    // (factual_alignment.py:302-352). A token that matches the next expected token
    // of an active lock (or a sequence start) MUST advance/start the lock — even if
    // it is also a helper token. The previous C++ checked helpers FIRST and returned
    // early, so it failed to advance the suffix, corrupting verbatim retrieval
    // (e.g. dropping a digit from a needle). Helper passthrough is the FALLBACK,
    // only when the token matched nothing.
    const auto& seqs       = srl_state.current_step_factual_sequences;
    const auto& entity_ids = srl_state.current_step_sequence_entity_ids;

    std::vector<std::vector<int32_t>> new_candidates;

    // 1. Try to advance existing active locks.
    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty() && suffix[0] == token_id) {
            new_candidates.emplace_back(suffix.begin() + 1, suffix.end());
        }
    }
    bool did_match_existing = !new_candidates.empty();

    bool has_active_lock = false;
    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty()) { has_active_lock = true; break; }
    }

    // 2. If nothing advanced and we are unlocked, try to START a new lock.
    bool did_start_new = false;
    if (!did_match_existing && !has_active_lock) {
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (!seqs[i].empty() && seqs[i][0] == token_id) {
                new_candidates.emplace_back(seqs[i].begin() + 1, seqs[i].end());
                did_start_new = true;
                int32_t seq_entity = (i < entity_ids.size()) ? entity_ids[i] : -1;
                if (seq_entity != -1) srl_state.current_entity_id = seq_entity;
            }
        }
    }

    // 3. Matched an existing candidate or started a new lock → commit.
    if (did_match_existing || did_start_new) {
        srl_state.vsl_active_candidates = new_candidates;
        srl_state.vsl_consecutive_helpers = 0;
        return;
    }

    // 4. Otherwise, a helper token passes through WITHOUT breaking the lock.
    if (helper_ids.count(token_id) > 0) {
        srl_state.vsl_consecutive_helpers++;
        if (srl_state.vsl_consecutive_helpers >= 12) {
            srl_state.vsl_active_candidates.clear();
        }
        return;
    }

    // 5. Non-helper token that matched nothing → the lock is broken.
    srl_state.vsl_active_candidates.clear();
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
