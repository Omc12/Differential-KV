// factual_alignment.hpp
#pragma once

#include "native_core/srl/session_srl_state.hpp"
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cctype>

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
        "can", "could", "will", "would", "shall", "should", "may", "might", "must", "say", "says", "said", "saying",
        "state", "states", "stated", "stating", "give", "gives", "given", "giving", "show", "shows", "shown", "showing",
        "write", "writes", "written", "writing", "read", "reads", "mention", "mentions", "mentioned", "describe", "describes",
        "described", "refer", "refers", "referred", "contain", "contains", "contained", "include", "includes", "included",
        "follow", "follows", "following", "followed", "find", "finds", "found", "express", "expresses", "expressed",
        "source", "document", "text", "passage", "file", "formula", "relation", "equation", "equations",
        "theorem", "definition", "fact", "facts", "retrieval", "information", "detail", "details", "exact", "exactly",
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
        for (const auto& seq : srl_state.current_step_factual_sequences) {
            for (int32_t t_id : seq) {
                allowed.insert(t_id);
            }
        }
    }
    
    return allowed;
}

inline void update_vsl_state_cpp(
    int32_t token_id,
    SessionSRLState& srl_state,
    const std::unordered_set<int32_t>& helper_ids
) {
    if (helper_ids.count(token_id) > 0) {
        srl_state.vsl_consecutive_helpers++;
        // Threshold lowered 6 → 4: with tight 5%-selection factual sequences, the model
        // should stay locked to a source phrase. 4 consecutive helpers signals real drift.
        if (srl_state.vsl_consecutive_helpers >= 4) {
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
        for (const auto& seq : srl_state.current_step_factual_sequences) {
            for (size_t j = 0; j < seq.size(); ++j) {
                if (seq[j] == token_id) {
                    std::vector<int32_t> next_suffix(seq.begin() + j + 1, seq.end());
                    new_candidates.push_back(next_suffix);
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
