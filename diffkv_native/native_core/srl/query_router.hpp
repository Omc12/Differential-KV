// query_router.hpp
// Pruned helper declarations for DiffKV query router state updates.
// (The dead header-only routing implementation has been deleted).

#pragma once

#include "native_core/srl/session_srl_state.hpp"
#include "native_core/srl/chunk_descriptor.hpp"
#include "native_core/srl/chunk_graph.hpp"
#include "native_core/srl/inverted_index.hpp"
#include "native_core/srl/semantic_index.hpp"

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cassert>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace diffkv {

// Out-of-line SRL state builders/helpers defined in query_router.cpp
void update_srl_from_compressed_block(
    SessionSRLState&               state,
    const float*                   desc_f32,
    int32_t                        slot_id,
    const int32_t*                 token_ids,
    int                            block_len,
    int                            start_pos,
    const std::unordered_set<int>& stop_tokens
);

SessionSRLState build_srl_state_from_blocks(
    const float*                   desc_matrix,
    const int32_t*                 slot_ids,
    int                            N,
    const int32_t*                 token_ids,
    int                            seq_len,
    int                            block_size,
    const std::unordered_set<int>& stop_tokens,
    int                            K_semantic         = 6,
    int                            K_temporal         = 2,
    float                          overlap_threshold  = 0.15f,
    bool                           add_first_as_sink  = true,
    bool                           add_last_as_sink   = true,
    const std::vector<int>*        block_anchor_idxs  = nullptr,
    int                            cached_len         = 0
);

std::string format_routing_stats(const SessionSRLState& state);

} // namespace diffkv
