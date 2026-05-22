"""
STAGE 2 — SAT: Sparse KV Residency Verifier
Phase 38.9 — Sparse Attention Transition

Tracks sparse KV cache lifecycle at block granularity:
  - sparse block write (first materialisation)
  - sparse block hit   (reuse without rematerialisation)
  - sparse block eviction
  - dense KV rematerialisation (a sparse block had to be rebuilt as dense)
  - cache reuse effectiveness (hits / (hits + misses))
  - sparse block continuity   (fraction of steps with no eviction storm)

All measurements reflect real cache events. Nothing is estimated.
"""

import time
import json
import threading
import os
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple


TRACE_PATH = "traces/stage2/phase_38_9_sat/kv_residency_trace.jsonl"


@dataclass
class KVBlockRecord:
    block_id: str
    layer_idx: int
    token_start: int
    token_end: int
    first_seen: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    eviction_count: int = 0
    rematerialisation_count: int = 0
    is_sparse: bool = True


class SparseKVResidencyVerifier:
    """
    STAGE 2 SAT: Sparse KV Residency Verifier.

    Wire this into the KV cache allocation / lookup logic so that
    every real cache event (write, hit, evict, rematerialise) is recorded.

    Example wiring in a KV cache manager:

        # On sparse block write:
        verifier.record_sparse_write(block_id, layer, t_start, t_end)

        # On cache hit:
        verifier.record_sparse_hit(block_id, layer)

        # On eviction:
        verifier.record_eviction(block_id, layer)

        # When a sparse block must be rebuilt as dense:
        verifier.record_dense_rematerialisation(block_id, layer, duration_ms)
    """

    def __init__(self, trace_path: str = TRACE_PATH, continuity_window: int = 100):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

        self._lock = threading.Lock()

        # Active block registry
        self._blocks: Dict[str, KVBlockRecord] = {}

        # Global counters
        self._total_sparse_writes = 0
        self._total_sparse_hits = 0
        self._total_evictions = 0
        self._total_rematerialisations = 0
        self._total_rematerialisation_ms = 0.0

        # Per-layer statistics
        self._layer_hits: Dict[int, int] = defaultdict(int)
        self._layer_remats: Dict[int, int] = defaultdict(int)

        # Sliding window for continuity score
        # Each slot is True if that step had NO eviction storm (≤ threshold evictions)
        self._continuity_window = continuity_window
        self._continuity_slots: deque = deque(maxlen=continuity_window)

        self._trace_buf: List[Dict] = []
        self._flush_every = 32
        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Core event recording
    # ------------------------------------------------------------------

    def record_sparse_write(
        self,
        block_id: str,
        layer_idx: int,
        token_start: int,
        token_end: int,
    ) -> None:
        """Record initial materialisation of a sparse KV block."""
        ts = time.time()
        with self._lock:
            self._total_sparse_writes += 1
            rec = KVBlockRecord(
                block_id=block_id,
                layer_idx=layer_idx,
                token_start=token_start,
                token_end=token_end,
                first_seen=ts,
                last_accessed=ts,
                is_sparse=True,
            )
            self._blocks[block_id] = rec
            self._emit("sparse_write", block_id, layer_idx, token_start, token_end, ts, {})

    def record_sparse_hit(self, block_id: str, layer_idx: int) -> None:
        """Record a cache hit — the sparse block was found and reused."""
        ts = time.time()
        with self._lock:
            self._total_sparse_hits += 1
            self._layer_hits[layer_idx] += 1
            if block_id in self._blocks:
                self._blocks[block_id].hit_count += 1
                self._blocks[block_id].last_accessed = ts
            self._emit("sparse_hit", block_id, layer_idx, None, None, ts, {})

    def record_eviction(self, block_id: str, layer_idx: int) -> None:
        """Record a block eviction from sparse KV cache."""
        ts = time.time()
        with self._lock:
            self._total_evictions += 1
            if block_id in self._blocks:
                self._blocks[block_id].eviction_count += 1
            self._emit("eviction", block_id, layer_idx, None, None, ts, {})
            # remove from active registry
            self._blocks.pop(block_id, None)

    def record_dense_rematerialisation(
        self,
        block_id: str,
        layer_idx: int,
        duration_ms: float,
        trigger: str = "unknown",
    ) -> None:
        """
        Record that a sparse block was rebuilt as a dense tensor.
        This is the key signal of remaining dense-tax.
        """
        ts = time.time()
        with self._lock:
            self._total_rematerialisations += 1
            self._total_rematerialisation_ms += duration_ms
            self._layer_remats[layer_idx] += 1
            if block_id in self._blocks:
                self._blocks[block_id].rematerialisation_count += 1
                self._blocks[block_id].is_sparse = False
            self._emit(
                "dense_rematerialisation", block_id, layer_idx, None, None, ts,
                {"duration_ms": round(duration_ms, 4), "trigger": trigger}
            )

    def mark_step_continuity(self, eviction_count_this_step: int, threshold: int = 4) -> None:
        """
        Call once per decode step with the number of evictions that occurred.
        Fills the sliding continuity window.
        """
        with self._lock:
            self._continuity_slots.append(eviction_count_this_step <= threshold)

    # ------------------------------------------------------------------
    # Metrics queries
    # ------------------------------------------------------------------

    def get_residency_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_lookups = self._total_sparse_hits + self._total_rematerialisations
            reuse_rate = self._total_sparse_hits / max(total_lookups, 1)
            remat_rate = self._total_rematerialisations / max(total_lookups, 1)

            continuity = (
                sum(self._continuity_slots) / len(self._continuity_slots)
                if self._continuity_slots else 1.0
            )

            return {
                "sparse_writes": self._total_sparse_writes,
                "sparse_hits": self._total_sparse_hits,
                "evictions": self._total_evictions,
                "dense_rematerialisations": self._total_rematerialisations,
                "total_rematerialisation_ms": round(self._total_rematerialisation_ms, 4),
                "avg_rematerialisation_ms": round(
                    self._total_rematerialisation_ms / max(self._total_rematerialisations, 1), 4
                ),
                "cache_reuse_rate": round(reuse_rate, 4),
                "rematerialisation_rate": round(remat_rate, 4),
                "sparse_block_continuity": round(continuity, 4),
                "active_sparse_blocks": len(self._blocks),
                "elapsed_sec": round(time.time() - self._session_start, 2),
            }

    def get_hot_remat_layers(self, top_n: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            sorted_layers = sorted(
                self._layer_remats.items(), key=lambda x: x[1], reverse=True
            )
            return [{"layer": li, "rematerialisations": cnt}
                    for li, cnt in sorted_layers[:top_n]]

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(
        self, event_type: str, block_id: str, layer_idx: int,
        t_start: Optional[int], t_end: Optional[int],
        ts: float, extra: Dict
    ) -> None:
        entry = {
            "ts": ts, "event": event_type, "block_id": block_id,
            "layer": layer_idx, "token_start": t_start, "token_end": t_end,
            **extra,
        }
        self._trace_buf.append(entry)
        if len(self._trace_buf) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._trace_buf:
            return
        with open(self.trace_path, "a") as f:
            for ev in self._trace_buf:
                f.write(json.dumps(ev) + "\n")
        self._trace_buf.clear()
