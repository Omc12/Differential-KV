#pragma once

#include <vector>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <ggml.h>

#include <functional>

#include "native_core/srl/inverted_index.hpp"

namespace dkv {

// RC3 — entity-signature dimensionality + deterministic generator.  Must match
// the Python factual_store.entity_signature formula exactly so that any future
// shared tooling agrees.
constexpr int ENTITY_SIG_DIM = 16;

inline std::vector<float> entity_signature(int32_t token_id, int dim = ENTITY_SIG_DIM) {
    std::vector<float> v(dim, 0.0f);
    if (token_id < 0) return v;
    double nrm = 0.0;
    for (int i = 0; i < dim; ++i) {
        double k = (double)(i + 1);
        double val = std::sin((double)token_id * 0.61803398875 * k + k);
        v[i] = (float)val;
        nrm += val * val;
    }
    nrm = std::sqrt(nrm);
    if (nrm > 1e-8) for (int i = 0; i < dim; ++i) v[i] = (float)(v[i] / nrm);
    return v;
}

inline bool is_gpt2_alnum(unsigned char b) {
    if (b <= 0x1f) return true;
    if (b >= 0x30 && b <= 0x39) return true;
    if (b >= 0x41 && b <= 0x5a) return true;
    if (b >= 0x61 && b <= 0x7a) return true;
    if (b >= 0x7f && b <= 0xa0) return true;
    if (b == 0xaa || b == 0xad) return true;
    if (b == 0xb2 || b == 0xb3 || b == 0xb5 || b == 0xb9 || b == 0xba) return true;
    if (b >= 0xbc && b <= 0xbe) return true;
    if (b >= 0xc0 && b <= 0xff) {
        return (b != 0xd7 && b != 0xf7);
    }
    return false;
}


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
    // Entity assignment: start_idx of the prime that owns this property span.
    // -1 = unassigned.  Set during build() by assign_entities() using a
    // distinguishing-token override plus nearest-preceding-prime (reading-order)
    // ownership — NOT plain token overlap (mis-binds shared-vocabulary
    // properties, RC4) and NOT absolute positional proximity (mis-binds
    // interleaved comparisons).
    int32_t entity_id = -1;
    // For prime entries only: the single highest-IDF (most distinguishing) token
    // in this prime's span, or -1.  A property span containing exactly one
    // prime's distinguishing token binds to that prime with high confidence.
    int32_t distinguishing_token = -1;
    // Provenance: start index of the ORIGINAL (pre-chunk) contiguous salient span
    // this entry was carved from. Chunks of one span share this value, letting
    // merge_adjacent_entries() rejoin a span that the 20-token chunker split
    // through (e.g. a digit run like "847291") even when the halves were promoted
    // to distinct prime entities. -1 = unset.
    int32_t orig_span_start = -1;
    // RC3: deterministic unit signature of this span's owning entity (from that
    // entity's distinguishing token); empty/zero when no owning entity.  Used
    // only when a query supplies an entity bias to rank the queried entity's
    // spans above shared-vocabulary spans of other entities.
    std::vector<float> entity_sig;
    mutable float current_sim = 0.0f;
    // 3-token source context preceding this span (RC2 quote-grounded connectives).
    std::vector<int32_t> prefix_tokens;
    // For prime entries: list of (bridge_tokens + value_tokens) sequences grounding
    // the complete (entity → relation → value) ordering in the source text (RC1).
    std::vector<std::vector<int32_t>> triple_sequences;
    // DX1: true when this span sits immediately after its prime via a definitional
    // bridge (copula IDF < 1.0 — "is", "are", "means", "refers to").
    bool is_definition = false;
    bool recalled = false;
};

class FactualExactStore {
public:
    std::string session_id;
    std::vector<FactEntry> entries;
    std::vector<float> eagle_scores;
    int num_layers = 0;
    int F_test = 0;
    float avg_r = 1.0f;
    // RC3: entity_id (prime start_idx) → that entity's distinguishing token.
    std::unordered_map<int32_t, int32_t> entity_distinguishing;
    std::unordered_map<int32_t, std::vector<int>> slot_to_entry_indices;

    FactualExactStore(const std::string& sess_id = "default") : session_id(sess_id), num_layers(0), F_test(0) {}

    void clear() {
        entries.clear();
        eagle_scores.clear();
        entity_distinguishing.clear();
        slot_to_entry_indices.clear();
        num_layers = 0;
        F_test = 0;
        avg_r = 1.0f;
    }

    void build(
        const std::vector<std::vector<ggml_fp16_t>>& k_activations, // [n_layers][L * kv_heads * head_dim] (fp16)
        const std::vector<std::vector<ggml_fp16_t>>& v_activations,
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
        const std::unordered_set<int32_t>& helper_token_ids,
        const std::unordered_set<int32_t>& relational_token_ids,
        std::function<std::string(int32_t)> token_to_piece_fn = nullptr,
        bool use_salience_parser = true
    );

    std::vector<FactEntry> query(
        const float* Q, // [H_q * head_dim]
        int H_q,
        int head_dim,
        const float* W_proj, // [DESC_DIM * head_dim]
        int desc_dim,
        float threshold = 0.4f,
        const std::unordered_set<int32_t>* active_slots = nullptr,
        const std::unordered_set<int32_t>* query_entity_bias = nullptr  // RC3
    ) const;
};

} // namespace dkv
