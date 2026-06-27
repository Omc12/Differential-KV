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
#include <iostream>

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
    std::string text_copy = text;
    size_t pos = 0;
    while ((pos = text_copy.find("</w>")) != std::string::npos) {
        text_copy.erase(pos, 4);
    }
    
    for (char c : text_copy) {
        unsigned char b = static_cast<unsigned char>(c);
        if (b == 0x20) continue;
        
        bool is_alnum = false;
        if (b <= 0x1f) is_alnum = true;
        else if (b >= 0x30 && b <= 0x39) is_alnum = true;
        else if (b >= 0x41 && b <= 0x5a) is_alnum = true;
        else if (b >= 0x61 && b <= 0x7a) is_alnum = true;
        else if (b >= 0x7f && b <= 0xa0) is_alnum = true;
        else if (b == 0xaa || b == 0xad) is_alnum = true;
        else if (b == 0xb2 || b == 0xb3 || b == 0xb5 || b == 0xb9 || b == 0xba) is_alnum = true;
        else if (b >= 0xbc && b <= 0xbe) is_alnum = true;
        else if (b >= 0xc0 && b <= 0xff) {
            is_alnum = (b != 0xd7 && b != 0xf7);
        }
        
        if (is_alnum) {
            if (b >= 'A' && b <= 'Z') {
                cleaned += static_cast<char>(b + 32);
            } else {
                cleaned += c;
            }
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
        // Pronouns & Determiners
        "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "that", "this", "these", "those", "each", "every", "some", "any", "no", "all", "both",
        "either", "neither", "another", "other", "such", "what", "a", "an", "the",
        // Prepositions & Conjunctions
        "of", "in", "to", "for", "with", "on", "at", "by", "from", "about", "as", "into", "through", "during", "before", "after",
        "above", "below", "up", "down", "over", "under", "between", "among", "out", "off", "within", "without", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "and", "or", "but", "so", "yet", "nor", "although",
        "because", "since", "unless", "until", "while", "whereas", "if", "else", "than",
        // Verbs (core + modal + reporting) — matching Python ACTIVE_RUNTIME ALLOWED_HELPER_WORDS
        "is", "was", "were", "are", "be", "been", "being", "am", "have", "has", "had", "having", "do", "does", "did", "doing",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must", "say", "says", "said", "saying",
        "state", "states", "stated", "stating", "give", "gives", "given", "giving", "show", "shows", "shown", "showing",
        "write", "writes", "written", "writing", "read", "reads", "mention", "mentions", "mentioned", "describe", "describes",
        "described", "refer", "refers", "referred", "contain", "contains", "contained", "include", "includes", "included",
        "follow", "follows", "following", "followed", "find", "finds", "found", "express", "expresses", "expressed",
        // Nouns & Adjectives (meta/structural) — matching Python ACTIVE_RUNTIME ALLOWED_HELPER_WORDS
        "source", "document", "text", "passage", "file", "formula", "relation", "equation", "equations",
        "theorem", "definition", "fact", "facts", "retrieval", "information", "detail", "details", "exact", "exactly",
        "correct", "correctly", "faithful", "faithfully", "verbatim", "missing", "present", "clear", "clearly",
        "uncertain", "certain", "provide", "provided", "provides", "note", "noted", "notes",
        // Common helper adverbs/adjectives
        "not", "only", "very", "too", "just", "well", "also", "now", "first", "second", "third", "one", "two", "three",
        // Mathematical and relational verbs/nouns
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

        // Build important-vocab-filtered recent set (matching Python ACTIVE_RUNTIME
        // diffkv_attention.py:857 — only high-IDF tokens count for anchor overlap
        // to avoid stopwords like "the"/"is" triggering the wrong entity).
        std::unordered_set<int32_t> recent_important;
        const auto& important_vocab = srl_state.inverted_index.important_vocab;
        for (int ri = rgt_start; ri < rgt_size; ++ri) {
            int32_t t = srl_state.recent_generated_tokens[ri];
            if (important_vocab.empty() || important_vocab.count(t)) {
                recent_important.insert(t);
            }
        }
        // Fall back to raw recent_set if important_vocab is empty (unbuilt)
        const std::unordered_set<int32_t>& effective_recent =
            important_vocab.empty() ? recent_set : recent_important;

        for (const FactEntry* fe : srl_state.prime_entries) {
            int overlap = 0;
            for (int32_t t : fe->tokens) {
                // Only count overlap on important (high-IDF) tokens
                if (effective_recent.count(t) &&
                    (important_vocab.empty() || important_vocab.count(t))) {
                    overlap++;
                }
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
            // Fix #5: Triple sequences don't appear in entries_by_tokens_map so they
            // score 0.0 and are evicted first — but they're the most factually valuable.
            // Inherit the owning prime's current_sim (matching Python ACTIVE_RUNTIME
            // diffkv_attention.py:946-955 "DX1: triple sequences inherit their prime's score").
            for (const FactEntry* fe : srl_state.prime_entries) {
                float prime_sim = fe->current_sim;
                if (prime_sim <= 0.0f) continue;
                for (const auto& triple_seq : fe->triple_sequences) {
                    for (size_t si = 0; si < n_seqs; ++si) {
                        if (seq_sims[si] == 0.0f &&
                            srl_state.current_step_factual_sequences[si] == triple_seq) {
                            seq_sims[si] = prime_sim;
                        }
                    }
                }
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

template <typename ModelType>
inline bool is_relational_connector_token(int32_t token_id, const ModelType& model) {
    static const std::unordered_set<std::string> RELATIONAL_WORDS = {
        "is", "are", "was", "were", "has", "have", "had",
        "exhibits", "possesses", "contains", "involves", "requires", "lacks", "features",
        "whereas", "while", "although", "but", "however", "yet", "though",
        "notwithstanding", "nevertheless", "nonetheless", "conversely",
        "unlike", "contrast", "instead", "rather",
        "because", "since", "therefore", "hence", "thus", "consequently",
        "accordingly", "than"
    };
    std::string piece = model.token_to_piece(token_id);
    std::string cleaned = clean_token_text(piece);
    if (cleaned.empty()) {
        for (char c : piece) {
            if (c == ':' || c == ';' || c == ',' || c == '.' || c == '=') {
                return true;
            }
        }
        return false;
    }
    return RELATIONAL_WORDS.count(cleaned) > 0;
}

// Returns the set of allowed token IDs for the current decode step.
// RC2: When SFA+lock active, relational binding words are stripped from helpers
// so they cannot leak between entity spans.  Pass structural_helper_ids (full
// helpers minus RELATIONAL_BINDING_WORDS) and sfa_active=true to activate this.
template <typename ModelType>
inline std::unordered_set<int32_t> get_allowed_tokens_vsl_cpp(
    const SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& helper_ids,
    const std::unordered_set<int32_t>* structural_helper_ids,
    bool sfa_active,
    const ModelType& model
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

        int32_t last_gen = -1;
        if (!srl_state.recent_generated_tokens.empty()) {
            last_gen = srl_state.recent_generated_tokens.back();
        }
        bool history_ends_in_connector = (last_gen >= 0 && is_relational_connector_token(last_gen, model));

        // Single pass: decide which sequence starts we may enter (entity-filtered)
        // and collect the source-adjacent connectives that bridge into them.
        std::unordered_set<int32_t> enterable_starts;
        std::unordered_set<int32_t> grounded_connectives;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (seqs[i].empty()) continue;
            int32_t seq_entity  = (i < entity_ids.size())    ? entity_ids[i]    : -1;
            bool    seq_is_prime = (i < is_prime_list.size()) ? is_prime_list[i] : false;

            bool enter = false;
            if (seq_is_prime || seq_entity == -1) {
                // Prime sequences always allowed; entity-agnostic sequences too
                // (matching Python ACTIVE_RUNTIME get_allowed_tokens_vsl lines 266-267)
                enter = true;
            } else if (current_entity != -1) {
                enter = (seq_entity == current_entity);
            } else if (dual_mode && !dual_ids.empty()) {
                enter = (dual_ids.count(seq_entity) > 0);
            } else {
                enter = true;
            }

            if (enter) {
                allowed.insert(seqs[i][0]);
                enterable_starts.insert(seqs[i][0]);

                // Also allow mid-sequence content tokens if they are valid entry points!
                for (size_t j = 1; j < seqs[i].size(); ++j) {
                    bool is_content = (helper_ids.count(seqs[i][j]) == 0);
                    if (is_content) {
                        bool prev_is_connector = is_relational_connector_token(seqs[i][j - 1], model);
                        if (prev_is_connector && history_ends_in_connector) {
                            allowed.insert(seqs[i][j]);
                        }
                    }
                }

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
inline std::string normalize_token_text(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = text[i];
        if (c == 0xC4 && i + 1 < text.size() && (unsigned char)text[i + 1] == 0x80) {
            i++; // skip 0x80
            continue;
        }
        if (c == '<' && i + 3 < text.size() && text.substr(i, 4) == "</w>") {
            i += 3;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '_') {
            continue;
        }
        if (c >= 'A' && c <= 'Z') {
            cleaned.push_back(c + ('a' - 'A'));
        } else {
            cleaned.push_back(c);
        }
    }
    return cleaned;
}

template <typename ModelType>
inline bool tokens_match_normalized(int32_t tok_a, int32_t tok_b, const ModelType& model) {
    if (tok_a == tok_b) return true;
    std::string str_a = normalize_token_text(model.token_to_piece(tok_a));
    std::string str_b = normalize_token_text(model.token_to_piece(tok_b));
    if (str_a.empty() || str_b.empty()) {
        return tok_a == tok_b;
    }
    return str_a == str_b;
}

// Advance the VSL lock state after a token is generated.
// Helpers pass through; threshold 4→12 so normal bridge phrases don't drop the lock.
// When starting a new lock from a sequence-start token, update current_entity_id
// from that sequence's entity tag so subsequent fallback steps are entity-consistent.
template <typename ModelType>
inline void update_vsl_state_cpp(
    int32_t token_id,
    SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& helper_ids,
    const ModelType& model
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
        if (!suffix.empty() && tokens_match_normalized(suffix[0], token_id, model)) {
            new_candidates.emplace_back(suffix.begin() + 1, suffix.end());
        }
    }
    bool did_match_existing = !new_candidates.empty();

    bool has_active_lock = false;
    for (const auto& suffix : srl_state.vsl_active_candidates) {
        if (!suffix.empty()) { has_active_lock = true; break; }
    }

    if (std::getenv("DIFFKV_VERBOSE")) {
        std::cerr << "[DiffKV VSL] Token: " << token_id 
                  << " | Candidates before: " << srl_state.vsl_active_candidates.size()
                  << " | Has active lock: " << has_active_lock
                  << "\n";
    }

    // 2. If nothing advanced and we are unlocked, try to START a new lock.
    // We allow starting a lock on any token inside a sequence if a suffix/substring
    // match is detected (meaning the generated history matches the prefix of that suffix).
    bool did_start_new = false;
    if (!did_match_existing && !has_active_lock) {
        for (size_t i = 0; i < seqs.size(); ++i) {
            const auto& seq = seqs[i];
            if (seq.empty()) continue;
            
            for (size_t j = 0; j < seq.size(); ++j) {
                if (tokens_match_normalized(seq[j], token_id, model)) {
                    // Compute match length of the prefix of seq[0..j] with recent_generated_tokens
                    int match_len = 1;
                    int recent_idx = (int)srl_state.recent_generated_tokens.size() - 1;
                    while (j >= (size_t)match_len && recent_idx >= 0) {
                        if (tokens_match_normalized(seq[j - match_len], srl_state.recent_generated_tokens[recent_idx], model)) {
                            match_len++;
                            recent_idx--;
                        } else {
                            break;
                        }
                    }
                    
                    // Standard lock starts at seq[0] with match_len=1.
                    // Enable mid-sequence lock starting when history match >= 5.
                    // Prevent locks starting on helper words at j = 0.
                    bool is_content_token = (helper_ids.count(token_id) == 0);
                    bool can_lock = (j == 0 && is_content_token) || 
                                    (j > 0 && is_content_token && match_len >= 2) || 
                                    (j > 0 && match_len >= 5);
                    if (can_lock) {
                        std::vector<int32_t> suffix(seq.begin() + j + 1, seq.end());
                        bool duplicate = false;
                        for (const auto& cand : new_candidates) {
                            if (cand == suffix) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            new_candidates.push_back(std::move(suffix));
                            did_start_new = true;
                            int32_t seq_entity = (i < entity_ids.size()) ? entity_ids[i] : -1;
                            if (seq_entity != -1) srl_state.current_entity_id = seq_entity;
                        }
                    }
                }
            }
        }
    }

    // 3. Matched an existing candidate or started a new lock → commit.
    if (did_match_existing || did_start_new) {
        bool completed = false;
        for (const auto& cand : new_candidates) {
            if (cand.empty()) {
                completed = true;
                break;
            }
        }
        if (completed) {
            for (auto& entry : srl_state.factual_store.entries) {
                if (entry.recalled) continue;
                int L = (int)entry.tokens.size();
                if (L >= 1 && tokens_match_normalized(entry.tokens[L - 1], token_id, model)) {
                    bool match = true;
                    int recent_size = (int)srl_state.recent_generated_tokens.size();
                    if (recent_size >= L - 1) {
                        for (int i = 0; i < L - 1; ++i) {
                            if (!tokens_match_normalized(srl_state.recent_generated_tokens[recent_size - (L - 1) + i], entry.tokens[i], model)) {
                                match = false;
                                break;
                            }
                        }
                    } else {
                        match = false;
                    }
                    if (match) {
                        entry.recalled = true;
                        if (std::getenv("DIFFKV_VERBOSE")) {
                            std::cerr << "[DiffKV VSL] Completed lock and marked entry as recalled: token_id=" << token_id << ", length=" << L << "\n";
                        }
                        // Propagate recall to larger/overlapping entries containing this sequence if it is long enough
                        if (L >= 5) {
                            for (auto& other : srl_state.factual_store.entries) {
                                if (other.recalled) continue;
                                if (other.tokens.size() > entry.tokens.size()) {
                                    bool found = false;
                                    for (size_t start = 0; start <= other.tokens.size() - entry.tokens.size(); ++start) {
                                        bool sub_match = true;
                                        for (size_t k = 0; k < entry.tokens.size(); ++k) {
                                            if (!tokens_match_normalized(other.tokens[start + k], entry.tokens[k], model)) {
                                                sub_match = false;
                                                break;
                                            }
                                        }
                                        if (sub_match) {
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (found) {
                                        other.recalled = true;
                                        if (std::getenv("DIFFKV_VERBOSE")) {
                                            std::cerr << "[DiffKV VSL] Propagated recall to overlapping entry: length=" << other.tokens.size() << "\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        srl_state.vsl_active_candidates = new_candidates;
        srl_state.vsl_consecutive_helpers = 0;
        if (std::getenv("DIFFKV_VERBOSE")) {
            std::cerr << "[DiffKV VSL] Committed lock: did_match_existing=" << did_match_existing 
                      << " did_start_new=" << did_start_new 
                      << " | Candidates after: " << srl_state.vsl_active_candidates.size() 
                      << "\n";
        }
        return;
    }

    // 4. Otherwise, a helper token passes through WITHOUT breaking the lock.
    if (helper_ids.count(token_id) > 0) {
        srl_state.vsl_consecutive_helpers++;
        if (std::getenv("DIFFKV_VERBOSE")) {
            std::cerr << "[DiffKV VSL] Helper passthrough: consecutive=" << srl_state.vsl_consecutive_helpers << "\n";
        }
        if (srl_state.vsl_consecutive_helpers >= 12) {
            srl_state.vsl_active_candidates.clear();
            if (std::getenv("DIFFKV_VERBOSE")) {
                std::cerr << "[DiffKV VSL] Helper threshold reached. Lock cleared.\n";
            }
        }
        return;
    }

    // 5. Non-helper token that matched nothing → the lock is broken.
    if (std::getenv("DIFFKV_VERBOSE")) {
        std::cerr << "[DiffKV VSL] Lock broken!\n";
    }
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
    const auto& structural_ids = get_structural_helper_token_ids_cpp(model);
    bool sfa_active = srl_state.current_step_max_similarity >= 0.40f &&
                      !srl_state.current_step_factual_sequences.empty();
    auto allowed = get_allowed_tokens_vsl_cpp(srl_state, helper_ids, &structural_ids, sfa_active, model);
    return allowed.count(token_id) > 0;
}

} // namespace diffkv
