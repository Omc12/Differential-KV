// diffkv_native/include/speculative.hpp
// Speculative decoding support for DiffKV.
//
// Translation of ACTIVE_RUNTIME/native_core/srl/speculative.py
//
// In the C++ llama.cpp context, speculative decoding works differently:
// - The main model is handled by llama.cpp inference
// - We track accepted/rejected tokens and rollback the KV cache
// - The SpeculativeDecodeState tracks in-flight candidates

#pragma once
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>

namespace diffkv {

// Result of one speculative decode step
struct SpeculativeStepResult {
    std::vector<int> accepted_tokens;   // tokens accepted from draft
    int correction_token = -1;          // correction token from main model (-1 if none)
    int target_len = 0;                 // target KV cache length after rollback
    
    std::vector<int> new_tokens() const {
        std::vector<int> all = accepted_tokens;
        if (correction_token >= 0) all.push_back(correction_token);
        return all;
    }
};

// Per-request speculative decode state
struct SpeculativeDecodeState {
    std::string session_id;
    int prefix_len = 0;         // KV cache length before this speculation step
    int num_candidates = 4;     // number of draft candidates to generate
    
    // Draft logits: [num_candidates][vocab_size]
    std::vector<std::vector<float>> draft_logits;
    std::vector<int> candidate_tokens;
    
    SpeculativeDecodeState() = default;
    SpeculativeDecodeState(const std::string& sid, int prefix, int n_cand = 4)
        : session_id(sid), prefix_len(prefix), num_candidates(n_cand) {}
};

// Greedy token from logits [vocab_size]
inline int greedy_sample(const float* logits, int vocab_size) {
    return (int)(std::max_element(logits, logits + vocab_size) - logits);
}

// Softmax over logits into probs
inline void softmax(const float* logits, float* probs, int vocab_size, float temperature) {
    float max_logit = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / std::max(temperature, 1e-5f));
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) probs[i] /= sum;
}

// Sample from probability distribution
inline int multinomial_sample(const float* probs, int vocab_size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return vocab_size - 1;
}

// Verify draft candidates against main model logits.
//
// main_logits: [len(verification_input), vocab_size] - main model outputs
// draft_logits_list: [num_candidates][vocab_size] - draft model outputs
// candidate_tokens: [num_candidates] - the drafted tokens
// prefix_len: KV cache length before this step
// temperature: sampling temperature (0.0 = greedy)
// stop_token_ids: set of stop token IDs
//
// Returns SpeculativeStepResult with accepted tokens, correction token,
// and the target KV length for rollback.
inline SpeculativeStepResult verify_candidates(
    const std::vector<std::vector<float>>& main_logits,     // [verify_len][vocab_size]
    const std::vector<std::vector<float>>& draft_logits,    // [num_candidates][vocab_size]
    const std::vector<int>& candidate_tokens,
    int prefix_len,
    float temperature,
    const std::vector<int>& stop_token_ids,
    std::mt19937& rng
) {
    SpeculativeStepResult result;
    int vocab_size = (int)main_logits[0].size();
    int num_candidates = (int)candidate_tokens.size();
    
    std::vector<bool> is_stop(vocab_size, false);
    for (int id : stop_token_ids) {
        if (id >= 0 && id < vocab_size) is_stop[id] = true;
    }
    
    for (int i = 0; i < num_candidates; ++i) {
        int target_token = candidate_tokens[i];
        const float* pred_logits = main_logits[i].data();
        
        if (temperature == 0.0f) {
            // Greedy verification
            int pred_token = greedy_sample(pred_logits, vocab_size);
            if (pred_token == target_token) {
                result.accepted_tokens.push_back(target_token);
                if (is_stop[target_token]) break;
            } else {
                result.correction_token = pred_token;
                break;
            }
        } else {
            // Sample-based verification (speculative rejection sampling)
            std::vector<float> main_probs(vocab_size), draft_probs(vocab_size);
            softmax(pred_logits, main_probs.data(), vocab_size, temperature);
            softmax(draft_logits[i].data(), draft_probs.data(), vocab_size, temperature);
            
            float p_main  = main_probs[target_token];
            float p_draft = draft_probs[target_token];
            
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng);
            
            if (r < std::min(1.0f, p_main / (p_draft + 1e-9f))) {
                result.accepted_tokens.push_back(target_token);
                if (is_stop[target_token]) break;
            } else {
                // Rejection: sample from normalized difference distribution
                std::vector<float> diff(vocab_size);
                float diff_sum = 0.0f;
                for (int j = 0; j < vocab_size; ++j) {
                    diff[j] = std::max(0.0f, main_probs[j] - draft_probs[j]);
                    diff_sum += diff[j];
                }
                if (diff_sum > 0.0f) {
                    for (int j = 0; j < vocab_size; ++j) diff[j] /= diff_sum;
                    result.correction_token = multinomial_sample(diff.data(), vocab_size, rng);
                } else {
                    result.correction_token = greedy_sample(pred_logits, vocab_size);
                }
                break;
            }
        }
    }
    
    // If all candidates accepted, sample correction from last logit
    if (result.correction_token < 0 &&
        (result.accepted_tokens.empty() ||
         !is_stop[result.accepted_tokens.back()])) {
        const float* last_logits = main_logits.back().data();
        if (temperature == 0.0f) {
            result.correction_token = greedy_sample(last_logits, vocab_size);
        } else {
            std::vector<float> probs(vocab_size);
            softmax(last_logits, probs.data(), vocab_size, temperature);
            result.correction_token = multinomial_sample(probs.data(), vocab_size, rng);
        }
    }
    
    // Target KV cache length after rollback
    result.target_len = prefix_len + (int)result.accepted_tokens.size();
    
    return result;
}

} // namespace diffkv
