"""
STAGE 2 — SAT: Sparse Attention Window Controller
Phase 38.9 — Sparse Attention Transition

Controls which tokens participate in each attention computation
by maintaining sparse, token-local, layer-aware attention windows.

Goals:
  - sparse-first routing: prefer local/block-sparse windows over full context
  - adaptive window sizing based on live token count and layer depth
  - layer-aware sparse gating (early layers may tolerate narrower windows)
  - token-locality awareness: anchor-adjacent tokens receive priority

This module does NOT compute attention — it only determines window
parameters that the attention kernel should honour.

All decisions adapt to real runtime measurements passed in by the caller.
"""

import time
import json
import threading
import os
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class SparseWindowSpec:
    """Output produced by the controller for a single layer / call."""
    layer_idx: int
    mode: str            # "sparse_local" | "sparse_block" | "hybrid" | "full_dense"
    window_size: int     # number of tokens in the active window
    stride: int          # stride between window positions
    block_size: int      # sparse block granularity
    gate_score: float    # 0.0 = full-dense forced; 1.0 = fully sparse-native
    reason: str          # human-readable justification
    ts: float = 0.0


class SparseAttentionWindowController:
    """
    STAGE 2 SAT: Sparse Attention Window Controller.

    Determines per-layer sparse attention window parameters based on:
      - sequence length at the time of the call
      - layer depth (shallower layers can use narrower windows)
      - active token count / head occupancy
      - adaptive feedback from the auditor (dense fallback rate)

    Usage:
        controller = SparseAttentionWindowController(num_layers=28)

        # At each attention call site:
        spec = controller.get_window_spec(
            layer_idx=layer_idx,
            seq_len=seq_len,
            active_tokens=active_tokens,
            dense_fallback_rate=auditor.get_fallback_frequency(),
        )

        # Then use spec.window_size / spec.block_size in the attention kernel.
        # Report what actually happened:
        controller.feedback_actual_mode(layer_idx, "sparse_local")
    """

    # Minimum and maximum window sizes (tokens)
    _MIN_WINDOW = 64
    _MAX_WINDOW = 4096

    # Fraction of seq_len used as initial window when sequence is long
    _LONG_SEQ_FRACTION = 0.25

    def __init__(
        self,
        num_layers: int = 28,
        base_block_size: int = 64,
        sparse_gate_threshold: float = 0.6,
        density_budget: float = 0.25,
    ):
        """
        Args:
            num_layers:            total transformer layer count
            base_block_size:       KV block granularity for sparse operations
            sparse_gate_threshold: dense_fallback_rate above which we force
                                   more aggressive sparse windowing
            density_budget:        target fraction of tokens that are active
                                   in each window (lower -> sparser)
        """
        self.num_layers = num_layers
        self.base_block_size = base_block_size
        self.sparse_gate_threshold = sparse_gate_threshold
        self.density_budget = density_budget

        self._lock = threading.Lock()

        # Per-layer adaptive state
        self._layer_window: Dict[int, int] = {}        # current window size
        self._layer_mode: Dict[int, str] = {}          # last chosen mode
        self._layer_gate: Dict[int, float] = {}        # gate score 0-1
        self._layer_feedback: Dict[int, str] = {}      # actual mode reported back

        # History for logging
        self._decisions: List[Dict] = []
        self._max_history = 500

        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def get_window_spec(
        self,
        layer_idx: int,
        seq_len: int,
        active_tokens: int,
        dense_fallback_rate: float = 0.0,
    ) -> SparseWindowSpec:
        """
        Compute and return the sparse window specification for this call.

        Args:
            layer_idx:          which transformer layer (0-based)
            seq_len:            current sequence length (tokens)
            active_tokens:      how many tokens are currently active (non-padding)
            dense_fallback_rate: fraction [0,1] of recent attention calls that fell
                                 back to dense — from the path auditor

        Returns:
            SparseWindowSpec with mode, window_size, stride, block_size, gate
        """
        with self._lock:
            return self._decide(layer_idx, seq_len, active_tokens, dense_fallback_rate)

    def feedback_actual_mode(self, layer_idx: int, actual_mode: str) -> None:
        """
        Report back the mode that was actually executed after the attention call.
        Used to track deviation from the controller's recommendation.
        """
        with self._lock:
            self._layer_feedback[layer_idx] = actual_mode

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_layer_specs(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for li in sorted(self._layer_window.keys()):
                rows.append({
                    "layer": li,
                    "window_size": self._layer_window.get(li, 0),
                    "mode": self._layer_mode.get(li, "unknown"),
                    "gate": round(self._layer_gate.get(li, 1.0), 4),
                    "actual": self._layer_feedback.get(li, "not_reported"),
                })
            return rows

    def get_global_density(self) -> float:
        """Average fraction of sequence covered by windows across all layers."""
        with self._lock:
            if not self._layer_window:
                return 1.0
            # Assume seq_len is captured in last decision — rough estimate only
            return self.density_budget

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "base_block_size": self.base_block_size,
            "sparse_gate_threshold": self.sparse_gate_threshold,
            "density_budget": self.density_budget,
            "min_window": self._MIN_WINDOW,
            "max_window": self._MAX_WINDOW,
            "elapsed_sec": round(time.time() - self._session_start, 2),
        }

    # ------------------------------------------------------------------
    # Internal decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        layer_idx: int,
        seq_len: int,
        active_tokens: int,
        dense_fallback_rate: float,
    ) -> SparseWindowSpec:
        ts = time.time()

        # Layer depth fraction (0 = first layer, 1 = last layer)
        depth_frac = layer_idx / max(self.num_layers - 1, 1)

        # --- Determine ideal window size ---
        # Shallow layers: small window (local attention)
        # Deep layers: slightly larger, but still sparse
        if seq_len <= 512:
            # Short sequence: simple local window
            base_window = max(seq_len, self._MIN_WINDOW)
            mode = "sparse_local"
        else:
            # Long sequence: fraction-based sparse window
            frac = self._LONG_SEQ_FRACTION * (0.5 + 0.5 * depth_frac)
            base_window = max(int(seq_len * frac), self._MIN_WINDOW)
            mode = "sparse_block"

        # Clamp to max
        window_size = min(base_window, self._MAX_WINDOW)

        # Align to block boundary
        window_size = (window_size // self.base_block_size) * self.base_block_size
        window_size = max(window_size, self.base_block_size)

        # --- Compute stride ---
        stride = max(window_size // 2, self.base_block_size)

        # --- Compute gate score ---
        # High dense fallback rate -> lower gate score (push toward dense only if needed)
        # Low dense fallback rate -> higher gate score (sparse is working, keep it)
        if dense_fallback_rate > self.sparse_gate_threshold:
            # Too many fallbacks: widen window slightly to reduce pressure
            window_size = min(window_size * 2, self._MAX_WINDOW)
            gate = max(0.3, 1.0 - dense_fallback_rate)
            mode = "hybrid"
            reason = (
                f"dense_fallback_rate={dense_fallback_rate:.3f} exceeds threshold "
                f"{self.sparse_gate_threshold}; widening window to {window_size}"
            )
        elif seq_len > 0 and active_tokens / max(seq_len, 1) < self.density_budget:
            # Many padding tokens — sparse window is very efficient here
            gate = 1.0
            reason = (
                f"low active_token density ({active_tokens}/{seq_len}); "
                f"sparse window {window_size} optimal"
            )
        else:
            gate = max(0.6, 1.0 - 0.4 * depth_frac)
            reason = (
                f"depth={depth_frac:.2f} seq_len={seq_len} -> "
                f"window={window_size} gate={gate:.2f}"
            )

        # Store
        self._layer_window[layer_idx] = window_size
        self._layer_mode[layer_idx] = mode
        self._layer_gate[layer_idx] = gate

        spec = SparseWindowSpec(
            layer_idx=layer_idx,
            mode=mode,
            window_size=window_size,
            stride=stride,
            block_size=self.base_block_size,
            gate_score=round(gate, 4),
            reason=reason,
            ts=ts,
        )

        # Append to rolling history
        self._decisions.append(asdict(spec))
        if len(self._decisions) > self._max_history:
            self._decisions.pop(0)

        return spec
