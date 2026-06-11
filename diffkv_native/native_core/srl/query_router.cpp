// query_router.cpp
// Compiled implementation units for the query router.
//
// Because the core algorithm is templated/inlined in query_router.hpp, this
// file provides:
//   1. Out-of-line explicit instantiation anchors (avoids symbol bloat)
//   2. The validate_srl_state() sanity-check utility
//   3. The update_srl_from_compressed_block() convenience wrapper used by
//      the ingest pipeline when a new block is appended.
//   4. The build_srl_state_from_blocks() batch-builder called at session
//      load / context restore.
//
// No PyTorch. Uses macOS Accelerate via the included headers.

#include "native_core/srl/query_router.hpp"
#include "native_core/srl/chunk_descriptor.hpp"
#include "native_core/srl/semantic_index.hpp"
#include "native_core/srl/inverted_index.hpp"
#include "native_core/srl/chunk_graph.hpp"
#include "native_core/srl/session_srl_state.hpp"

#include <iostream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <numeric>

namespace diffkv {

// ---------------------------------------------------------------------------
// validate_srl_state
//
// Lightweight consistency check — logs warnings to stderr.
// Returns true if state looks healthy.
// ---------------------------------------------------------------------------
bool validate_srl_state(const SessionSRLState& state, bool verbose) {
    bool ok = true;

    int N = state.n_active_blocks();

    if (state.semantic_index.N != N) {
        if (verbose) {
            std::cerr << "[SRL] WARNING: semantic_index.N=" << state.semantic_index.N
                      << " != ordered_slot_ids.size()=" << N << "\n";
        }
        ok = false;
    }

    if (static_cast<int>(state.semantic_index.desc_matrix.size()) != N * DESC_DIM) {
        if (verbose) {
            std::cerr << "[SRL] WARNING: desc_matrix size mismatch: "
                      << state.semantic_index.desc_matrix.size()
                      << " expected " << N * DESC_DIM << "\n";
        }
        ok = false;
    }

    if (state.chunk_graph.N != 0 && state.chunk_graph.N != N) {
        if (verbose) {
            std::cerr << "[SRL] WARNING: chunk_graph.N=" << state.chunk_graph.N
                      << " != N=" << N << "\n";
        }
        ok = false;
    }

    if (verbose && ok) {
        std::cerr << "[SRL] State OK: N=" << N
                  << " k_multiplier=" << state.k_multiplier
                  << " miss_rate=" << state.recent_miss_rate << "\n";
    }

    return ok;
}

// ---------------------------------------------------------------------------
// update_srl_from_compressed_block
//
// Called by the ingest pipeline each time a new block is compressed and
// assigned a pool slot.
//
// Parameters:
//   state      : the session SRL state to update (mutated in place)
//   desc_f32   : [DESC_DIM] float32, L2-normalised block descriptor
//   slot_id    : pool slot ID of the new block
//   token_ids  : [block_len] raw token IDs for lexical indexing (may be null)
//   block_len  : number of tokens in this block (0 if token_ids is null)
//   start_pos  : absolute start position of first token in this block
//   stop_tokens: stop token set for inverted-index building
// ---------------------------------------------------------------------------
void update_srl_from_compressed_block(
    SessionSRLState&               state,
    const float*                   desc_f32,
    int32_t                        slot_id,
    const int32_t*                 token_ids,
    int                            block_len,
    int                            start_pos,
    const std::unordered_set<int>& stop_tokens
) {
    // 1. Append to ordered slot list and semantic index
    state.ordered_slot_ids.push_back(slot_id);
    add_block_to_index(state.semantic_index, desc_f32, slot_id);

    // 2. Update inverted token index
    if (token_ids && block_len > 0) {
        // Add to chunk_vocabularies and occurrences
        std::vector<int32_t> toks(token_ids, token_ids + block_len);
        // We replicate InvertedIndex::add_block_tokens logic inline here
        auto& inv = state.inverted_index;
        for (int rel = 0; rel < block_len; ++rel) {
            int tok = static_cast<int>(toks[rel]);
            if (stop_tokens.count(tok)) continue;
            int abs_pos = start_pos + rel;
            inv.occurrences[tok].emplace_back(slot_id, abs_pos, rel);
            inv.chunk_vocabularies[slot_id][tok].push_back(rel);
        }
        // Recompute IDF
        int N_blocks = state.n_active_blocks();
        for (const auto& kv : inv.occurrences) {
            int tok = kv.first;
            std::unordered_set<int32_t> slots_with;
            for (const auto& occ : kv.second)
                slots_with.insert(std::get<0>(occ));
            float n_cont  = static_cast<float>(slots_with.size());
            float n_blk   = static_cast<float>(N_blocks);
            inv.idf[tok]  = std::log(n_blk / n_cont) + 1.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// build_srl_state_from_blocks
//
// Batch-initialise a SessionSRLState from a complete set of blocks.
// Used at session load or context restore.
//
// desc_matrix : [N * DESC_DIM] float32, L2-normalised (one row per block)
// slot_ids    : [N] pool slot IDs in chronological order
// token_ids   : [seq_len] complete token sequence (may be null)
// seq_len     : total token count
// block_size  : tokens per block
// stop_tokens : stop token set
//
// Also builds chunk graph and inverted index from scratch.
// ---------------------------------------------------------------------------
SessionSRLState build_srl_state_from_blocks(
    const float*                   desc_matrix,
    const int32_t*                 slot_ids,
    int                            N,
    const int32_t*                 token_ids,
    int                            seq_len,
    int                            block_size,
    const std::unordered_set<int>& stop_tokens,
    // Graph build params
    int    K_semantic         = 6,
    int    K_temporal         = 2,
    float  overlap_threshold  = 0.15f,
    // Sink blocks: first and last block slots
    bool   add_first_as_sink  = true,
    bool   add_last_as_sink   = true
) {
    SessionSRLState state;
    if (N == 0) return state;

    // --- Semantic index ---
    state.semantic_index = build_semantic_index(desc_matrix, slot_ids, N);

    // --- Ordered slot IDs ---
    state.ordered_slot_ids.assign(slot_ids, slot_ids + N);

    // --- Sink blocks ---
    if (add_first_as_sink && N > 0)
        state.sink_blocks.push_back(slot_ids[0]);
    if (add_last_as_sink && N > 1)
        state.sink_blocks.push_back(slot_ids[N - 1]);
    // Deduplicate sinks
    std::sort(state.sink_blocks.begin(), state.sink_blocks.end());
    state.sink_blocks.erase(
        std::unique(state.sink_blocks.begin(), state.sink_blocks.end()),
        state.sink_blocks.end());

    // --- Inverted index ---
    if (token_ids && seq_len > 0) {
        state.inverted_index = build_inverted_index(
            token_ids, seq_len,
            state.ordered_slot_ids,
            block_size,
            stop_tokens,
            /*top_n_per_block=*/20);
    }

    // --- Chunk graph ---
    state.chunk_graph = build_chunk_graph(
        desc_matrix,
        slot_ids,
        N,
        K_semantic,
        K_temporal,
        (token_ids && seq_len > 0) ? &state.inverted_index : nullptr,
        overlap_threshold);

    return state;
}

// ---------------------------------------------------------------------------
// format_routing_stats
//
// Returns a human-readable summary of the current routing state.
// ---------------------------------------------------------------------------
std::string format_routing_stats(const SessionSRLState& state) {
    std::ostringstream oss;
    oss << "SRL[N=" << state.n_active_blocks()
        << " k_mult=" << state.k_multiplier
        << " miss=" << state.recent_miss_rate
        << " calls=" << state.call_count
        << " last_k=" << state.current_step_count << "]";
    return oss.str();
}

} // namespace diffkv
