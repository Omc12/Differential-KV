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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <numeric>

namespace diffkv {

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

    // --- Per-decode-step cached slot selection ---
    std::vector<int32_t> current_step_slots;
    int current_step_count = 0;

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
        b.k_semantic = static_cast<int>(std::round(K * k_semantic_frac));
        b.k_lexical  = static_cast<int>(std::round(K * k_lexical_frac));
        b.k_graph    = static_cast<int>(std::round(K * k_graph_frac));
        b.k_recency  = static_cast<int>(std::round(K * k_recency_frac));
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
};

} // namespace diffkv
