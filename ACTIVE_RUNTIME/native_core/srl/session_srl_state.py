"""
native_core/srl/session_srl_state.py

Per-session SRL state: holds all routing indexes and running stats.
One instance per session, attached to KVRuntimeManager._session_srl[session_id].

Memory footprint for a 25K-token session (781 blocks):
  desc_matrix:   781 × 64 × 2 B  =  100 KB  (GPU)
  chunk_graph:   781 × 8  × 4 B  =   25 KB  (CPU/GPU)
  inverted_index: ~5000 terms × avg 12 blocks × 4 B = 240 KB  (CPU)
  ordered_slot_ids: 781 × 4 B    =    3 KB  (CPU)
  Total: ~368 KB  (vs ~400 MB compressed KV pool → <0.1% overhead)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from native_core.srl.semantic_index import SemanticIndex
from native_core.srl.chunk_graph import ChunkGraph
from native_core.srl.inverted_index import InvertedTokenIndex


@dataclass
class SessionSRLState:
    """All routing indexes for one session."""

    semantic_index:     SemanticIndex
    chunk_graph:        ChunkGraph
    inverted_index:     InvertedTokenIndex

    # Pool slot IDs in chronological order (one per block, layer-0 reference)
    ordered_slot_ids:   List[int]

    # Pool slot IDs that are always included (block 0, special-token blocks)
    sink_blocks:        List[int]

    # Adaptive-K running state
    recent_miss_rate:   float = 0.0
    k_multiplier:       float = 1.0
    call_count:         int   = 0

    # Rolling window of recently generated token IDs for lexical lookup
    recent_generated_tokens: List[int] = field(default_factory=list)

    # Per-decode-step cached slot selection (shared across all 28 layers)
    # Set by layer 0, consumed by layers 1–27
    current_step_slots: Optional[torch.Tensor] = None
    current_step_count: int = 0   # decode step counter for validation

    # Per-layer ordered slots (all layers share pool slots → same list for all)
    # Maps layer_idx → ordered list of pool slot IDs (currently identical for every layer)
    layer_slot_ids: Dict[int, List[int]] = field(default_factory=dict)

    # ── Config (all overridable via SRL_CONFIG or env vars) ──────────────────
    k_min:             int   = 20
    k_max:             int   = 200
    k_semantic_frac:   float = 0.50
    k_lexical_frac:    float = 0.15
    k_graph_frac:      float = 0.15
    k_recency_frac:    float = 0.20
    routing_threshold: int   = 50

    def n_active_blocks(self) -> int:
        """Number of compressed blocks in this session."""
        return len(self.ordered_slot_ids)

    def update_generated_tokens(self, token_id: int, maxlen: int = 64) -> None:
        """Append a newly generated token ID to the rolling window."""
        self.recent_generated_tokens.append(token_id)
        if len(self.recent_generated_tokens) > maxlen:
            self.recent_generated_tokens = self.recent_generated_tokens[-maxlen:]

    def update_miss_rate(self, attention_weights_k: torch.Tensor) -> None:
        """
        Update the exponential moving average miss-rate signal.

        A flat attention distribution over the selected blocks (max_weight ≈ 1/K)
        suggests the truly-relevant block wasn't selected → high miss signal.
        A peaked distribution (one block dominated) → low miss signal.
        """
        K = attention_weights_k.shape[0]
        if K == 0:
            return
        max_w = float(attention_weights_k.max())
        expected_max = 1.0 / K
        denom = (1.0 - expected_max) + 1e-8
        miss_signal = max(0.0, min(1.0, 1.0 - (max_w - expected_max) / denom))

        alpha = 0.05  # EMA smoothing
        self.recent_miss_rate = (1.0 - alpha) * self.recent_miss_rate + alpha * miss_signal

        # Boost K multiplier when miss rate is high
        if self.recent_miss_rate > 0.4:
            self.k_multiplier = min(self.k_multiplier * 1.2, 3.0)
        else:
            self.k_multiplier = max(self.k_multiplier * 0.99, 1.0)
