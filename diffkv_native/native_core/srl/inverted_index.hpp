// inverted_index.hpp
// Translation of inverted_index.py to C++17.
// Token-level inverted index for lexical retrieval over compressed KV blocks.

#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <functional>
#include <string>
#include <cstdlib>

namespace diffkv {

// ---------------------------------------------------------------------------
// InvertedTokenIndex
// ---------------------------------------------------------------------------
struct InvertedTokenIndex {
    // index: token_id -> sorted list of pool slot IDs containing that token
    std::unordered_map<int, std::vector<int32_t>> index;

    // important_vocab: high-IDF tokens identified during building
    std::unordered_set<int> important_vocab;

    // occurrences: token_id -> list of (slot_id, abs_pos, rel_pos)
    std::unordered_map<int, std::vector<std::tuple<int32_t, int, int>>> occurrences;

    // chunk_vocabularies: slot_id -> { token_id -> [rel_pos, ...] }
    std::unordered_map<int32_t,
        std::unordered_map<int, std::vector<int>>> chunk_vocabularies;

    // idf: token_id -> IDF score  (log(N_blocks / n_containing) + 1.0)
    std::unordered_map<int, float> idf;

    int block_size = 256;

    InvertedTokenIndex() : block_size(256) {}

    void add_block_tokens(int32_t slot_id, const std::vector<int32_t>& tokens, int32_t start_pos, const std::unordered_set<int32_t>& stop_tokens) {
        for (size_t rel_pos = 0; rel_pos < tokens.size(); ++rel_pos) {
            int32_t tok = tokens[rel_pos];
            if (stop_tokens.count(tok)) {
                continue;
            }
            int32_t abs_pos = start_pos + rel_pos;
            occurrences[tok].push_back({slot_id, abs_pos, (int32_t)rel_pos});
            chunk_vocabularies[slot_id][tok].push_back((int32_t)rel_pos);
        }
    }

    void recompute_idf(int32_t total_blocks) {
        idf.clear();
        float n_blocks = std::max(1.0f, (float)total_blocks);
        for (const auto& pair : occurrences) {
            int32_t tok = pair.first;
            std::unordered_set<int32_t> slots_containing;
            for (const auto& occ : pair.second) {
                slots_containing.insert(std::get<0>(occ));
            }
            float n_containing = (float)slots_containing.size();
            idf[tok] = std::log(n_blocks / n_containing) + 1.0f;
        }
    }

    std::vector<int32_t> search(const std::vector<int32_t>& query_tokens, int k_lexical) {
        if (query_tokens.empty() || occurrences.empty()) {
            return {};
        }

        float decay_factor = 0.999f;
        if (const char* env = std::getenv("DIFFKV_SRL_DECAY_FACTOR")) {
            decay_factor = std::stof(env);
        }

        int32_t L = 0;
        bool found_any = false;
        for (int32_t tok : query_tokens) {
            auto it = occurrences.find(tok);
            if (it != occurrences.end()) {
                for (const auto& occ : it->second) {
                    if (std::get<1>(occ) > L) {
                        L = std::get<1>(occ);
                    }
                    found_any = true;
                }
            }
        }

        if (!found_any) {
            return {};
        }

        std::unordered_map<int32_t, double> slot_scores;
        std::unordered_map<int32_t, std::unordered_set<int32_t>> slot_matched_toks;

        for (int32_t tok : query_tokens) {
            auto it = occurrences.find(tok);
            if (it != occurrences.end()) {
                float idf_val = 1.0f;
                auto idf_it = idf.find(tok);
                if (idf_it != idf.end()) {
                    idf_val = idf_it->second;
                }

                for (const auto& occ : it->second) {
                    slot_scores[std::get<0>(occ)] += idf_val * std::pow(decay_factor, L - std::get<1>(occ));
                    slot_matched_toks[std::get<0>(occ)].insert(tok);
                }
            }
        }

        std::vector<std::pair<int32_t, double>> slot_score_pairs;
        for (const auto& pair : slot_scores) {
            int32_t slot = pair.first;
            double score = pair.second;
            int32_t n_unique = slot_matched_toks[slot].size();
            score *= (n_unique * n_unique);
            slot_score_pairs.push_back({slot, score});
        }

        std::sort(slot_score_pairs.begin(), slot_score_pairs.end(), [](const auto& a, const auto& b) {
            if (std::abs(a.second - b.second) > 1e-9) {
                return a.second > b.second;
            }
            return a.first < b.first;
        });

        std::vector<int32_t> results;
        for (int i = 0; i < std::min(k_lexical, (int)slot_score_pairs.size()); ++i) {
            results.push_back(slot_score_pairs[i].first);
        }
        return results;
    }

    void clear() {
        index.clear();
        occurrences.clear();
        chunk_vocabularies.clear();
        idf.clear();
    }
};

using InvertedIndex = InvertedTokenIndex;

// ---------------------------------------------------------------------------
// build_inverted_index
// ---------------------------------------------------------------------------
inline InvertedTokenIndex build_inverted_index(
    const int32_t*                  token_ids,     // [seq_len]
    int                             seq_len,
    const std::vector<int32_t>&     slot_ids,      // [N] in chronological order
    int                             block_size,
    const std::unordered_set<int>&  stop_token_ids,
    int                             top_n_per_block = 20
) {
    InvertedTokenIndex inv;
    inv.block_size = block_size;

    int N = static_cast<int>(slot_ids.size());
    if (N == 0 || seq_len == 0) return inv;

    for (int i = 0; i < N; ++i) {
        int32_t slot = slot_ids[i];
        int tok_start = i * block_size;
        int tok_end   = std::min(tok_start + block_size, seq_len);
        if (tok_start >= seq_len) break;

        std::unordered_map<int, int> freq;
        for (int pos = tok_start; pos < tok_end; ++pos) {
            int tok = static_cast<int>(token_ids[pos]);
            if (!stop_token_ids.count(tok)) {
                freq[tok]++;
            }
        }

        std::vector<std::pair<int, int>> freq_vec(freq.begin(), freq.end());
        int n_top = std::min(top_n_per_block, static_cast<int>(freq_vec.size()));
        std::partial_sort(freq_vec.begin(), freq_vec.begin() + n_top, freq_vec.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

        for (int j = 0; j < n_top; ++j) {
            int tok = freq_vec[j].first;
            inv.index[tok].push_back(slot);
        }

        for (int pos = tok_start; pos < tok_end; ++pos) {
            int tok = static_cast<int>(token_ids[pos]);
            if (stop_token_ids.count(tok)) continue;
            int rel_pos = pos - tok_start;
            inv.occurrences[tok].emplace_back(slot, pos, rel_pos);
            inv.chunk_vocabularies[slot][tok].push_back(rel_pos);
        }
    }

    for (auto& kv : inv.index) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end());
        vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    }

    for (const auto& kv : inv.occurrences) {
        int tok = kv.first;
        std::unordered_set<int32_t> slots_with_tok;
        for (const auto& occ : kv.second)
            slots_with_tok.insert(std::get<0>(occ));
        float n_containing = static_cast<float>(slots_with_tok.size());
        float n_blocks     = static_cast<float>(N);
        inv.idf[tok] = std::log(n_blocks / n_containing) + 1.0f;
    }

    return inv;
}

// ---------------------------------------------------------------------------
// lookup
// ---------------------------------------------------------------------------
inline std::unordered_set<int32_t> lookup(
    const InvertedTokenIndex&   inv_index,
    const std::vector<int>&     query_token_ids
) {
    std::unordered_set<int32_t> result;
    for (int tok : query_token_ids) {
        auto it = inv_index.index.find(tok);
        if (it != inv_index.index.end()) {
            for (int32_t slot : it->second)
                result.insert(slot);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// score_lexical_slots
// ---------------------------------------------------------------------------
inline std::vector<std::pair<int32_t, float>> score_lexical_slots(
    const InvertedTokenIndex&   inv_index,
    const std::vector<int>&     query_token_ids,
    float                       decay = 0.999f
) {
    if (query_token_ids.empty() || inv_index.occurrences.empty())
        return {};

    int L = 0;
    bool found_any = false;
    for (int tok : query_token_ids) {
        auto it = inv_index.occurrences.find(tok);
        if (it != inv_index.occurrences.end()) {
            for (const auto& occ : it->second) {
                int abs_pos = std::get<1>(occ);
                if (abs_pos > L) L = abs_pos;
                found_any = true;
            }
        }
    }
    if (!found_any) return {};

    std::unordered_map<int32_t, double> slot_scores;
    std::unordered_map<int32_t, std::unordered_set<int>> slot_matched;

    for (int tok : query_token_ids) {
        auto occ_it = inv_index.occurrences.find(tok);
        if (occ_it == inv_index.occurrences.end()) continue;

        float idf_val = 1.0f;
        auto idf_it = inv_index.idf.find(tok);
        if (idf_it != inv_index.idf.end()) idf_val = idf_it->second;

        for (const auto& occ : occ_it->second) {
            int32_t slot   = std::get<0>(occ);
            int     abs_pos = std::get<1>(occ);
            double  w       = idf_val * std::pow(static_cast<double>(decay),
                                                  static_cast<double>(L - abs_pos));
            slot_scores[slot] += w;
            slot_matched[slot].insert(tok);
        }
    }

    std::vector<std::pair<int32_t, float>> result;
    result.reserve(slot_scores.size());
    for (const auto& kv : slot_scores) {
        int32_t slot    = kv.first;
        double  score   = kv.second;
        double  n_uniq  = static_cast<double>(slot_matched[slot].size());
        score *= n_uniq * n_uniq;
        result.emplace_back(slot, static_cast<float>(score));
    }

    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return result;
}

// ---------------------------------------------------------------------------
// compute_block_overlap
// ---------------------------------------------------------------------------
inline float compute_block_overlap(
    const std::unordered_map<int, std::vector<int>>& vocab_a,
    const std::unordered_map<int, std::vector<int>>& vocab_b
) {
    if (vocab_a.empty() || vocab_b.empty()) return 0.0f;

    int intersect = 0;
    for (const auto& kv : vocab_a) {
        if (vocab_b.count(kv.first)) ++intersect;
    }
    int union_sz = static_cast<int>(vocab_a.size() + vocab_b.size()) - intersect;
    if (union_sz <= 0) return 0.0f;
    return static_cast<float>(intersect) / static_cast<float>(union_sz);
}

} // namespace diffkv
