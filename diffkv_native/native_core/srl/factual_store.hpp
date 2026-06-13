#pragma once

#include <vector>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>

#include "native_core/srl/inverted_index.hpp"

namespace diffkv {

struct FactEntry {
    int start_idx;
    int end_idx;
    // Keys & Values across all layers:
    // shape: [num_layers, kv_heads, span_len, head_dim]
    // Flat vector of size: num_layers * kv_heads * span_len * head_dim
    std::vector<float> K; 
    std::vector<float> V;
    std::vector<float> descriptor; // [DESC_DIM]
    std::vector<int32_t> slot_ids; // slot IDs this fact belongs to
    std::vector<int32_t> tokens;   // token IDs covered by this span
    std::vector<int> neighbors;    // indices of connected factual entries
    std::vector<float> weights;    // connection weights
    bool is_prime = false;
    mutable float current_sim = 0.0f;
};

class FactualExactStore {
public:
    std::string session_id;
    std::vector<FactEntry> entries;
    int num_layers = 0;
    int F_test = 0;

    FactualExactStore(const std::string& sess_id = "default") : session_id(sess_id), num_layers(0), F_test(0) {}

    void clear() {
        entries.clear();
        num_layers = 0;
        F_test = 0;
    }

    void build(
        const std::vector<std::vector<float>>& k_activations, // [n_layers][L * kv_heads * head_dim]
        const std::vector<std::vector<float>>& v_activations,
        const std::vector<int32_t>& token_ids,
        const float* W_proj, // [DESC_DIM * head_dim]
        int desc_dim,
        int head_dim,
        int kv_heads,
        const std::unordered_set<int>& stop_token_ids,
        const std::vector<int32_t>& slot_ids, // active slot IDs in chronological order
        int block_size,
        const InvertedTokenIndex& inv_index,
        const std::unordered_set<int32_t>& semantic_prime_slots,
        bool use_salience_parser = true
    );

    std::vector<FactEntry> query(
        const float* Q, // [H_q * head_dim]
        int H_q,
        int head_dim,
        const float* W_proj, // [DESC_DIM * head_dim]
        int desc_dim,
        float threshold = 0.4f,
        const std::unordered_set<int32_t>* active_slots = nullptr
    ) const;
};

} // namespace diffkv
