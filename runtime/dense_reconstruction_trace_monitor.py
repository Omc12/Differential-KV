"""
STAGE 2 — SAT: Dense Reconstruction Trace Monitor
Phase 38.9 — Sparse Attention Transition

Records every dense KV reconstruction event with:
  - wall-clock timestamp and duration
  - triggering condition (e.g. cache miss, attention fallback, sequence extension)
  - affected layer(s)
  - affected token range
  - cumulative reconstruction pressure across the session

Purpose: identify remaining hidden dense-tax that prevents the
transformer from operating in a fully sparse-native execution regime.

All records derive from real events. Zero synthetic injection.
"""

import time
import json
import threading
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any


TRACE_PATH = "traces/stage2/phase_38_9_sat/dense_reconstruction_trace.jsonl"


TRIGGER_CACHE_MISS         = "cache_miss"
TRIGGER_ATTENTION_FALLBACK = "attention_fallback"
TRIGGER_SEQUENCE_EXTENSION = "sequence_extension"
TRIGGER_FORCED_DENSE       = "forced_dense_path"
TRIGGER_CAPACITY_EXCEEDED  = "sparse_capacity_exceeded"
TRIGGER_UNKNOWN            = "unknown"


@dataclass
class ReconstructionEvent:
    event_id: int
    timestamp: float
    duration_ms: float
    trigger: str
    layer_indices: List[int]
    token_start: int
    token_end: int
    token_count: int
    cumulative_events: int
    cumulative_ms: float
    metadata: Dict = field(default_factory=dict)


class DenseReconstructionTraceMonitor:
    """
    STAGE 2 SAT: Dense Reconstruction Trace Monitor.

    Must be wired into actual KV cache / attention code paths where
    dense reconstruction is triggered.  Call `record_reconstruction()`
    at the exact point where a dense materialisation begins and ends.

    Usage (inline):
        t0 = time.perf_counter()
        # ... actual dense reconstruction work ...
        t1 = time.perf_counter()
        monitor.record_reconstruction(
            trigger=TRIGGER_CACHE_MISS,
            layer_indices=[4, 5],
            token_start=128,
            token_end=256,
            duration_ms=(t1 - t0) * 1000,
        )

    Usage (context manager):
        with monitor.trace_reconstruction(TRIGGER_ATTENTION_FALLBACK, [3]) as ctx:
            ctx.set_token_range(64, 192)
            # ... dense reconstruction ...
    """

    def __init__(self, trace_path: str = TRACE_PATH):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

        self._lock = threading.Lock()
        self._event_counter = 0
        self._cumulative_ms = 0.0

        # Per-trigger histograms
        self._trigger_counts: Dict[str, int] = {}
        self._trigger_ms: Dict[str, float] = {}

        # Per-layer exposure count
        self._layer_exposures: Dict[int, int] = {}

        self._trace_buf: List[Dict] = []
        self._flush_every = 16
        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Primary recording API
    # ------------------------------------------------------------------

    def record_reconstruction(
        self,
        trigger: str,
        layer_indices: List[int],
        token_start: int,
        token_end: int,
        duration_ms: float,
        metadata: Optional[Dict] = None,
    ) -> ReconstructionEvent:
        """
        Record a single dense reconstruction event from real execution.

        Args:
            trigger:       one of the TRIGGER_* constants (or a custom string)
            layer_indices: which transformer layers were affected
            token_start:   first token position reconstructed
            token_end:     last  token position reconstructed (exclusive)
            duration_ms:   wall-clock cost of the reconstruction
            metadata:      optional free-form dict for extra context
        """
        ts = time.time()
        token_count = max(token_end - token_start, 0)

        with self._lock:
            self._event_counter += 1
            self._cumulative_ms += duration_ms

            # Aggregate per-trigger
            self._trigger_counts[trigger] = self._trigger_counts.get(trigger, 0) + 1
            self._trigger_ms[trigger] = self._trigger_ms.get(trigger, 0.0) + duration_ms

            # Aggregate per-layer
            for li in layer_indices:
                self._layer_exposures[li] = self._layer_exposures.get(li, 0) + 1

            ev = ReconstructionEvent(
                event_id=self._event_counter,
                timestamp=ts,
                duration_ms=round(duration_ms, 4),
                trigger=trigger,
                layer_indices=layer_indices,
                token_start=token_start,
                token_end=token_end,
                token_count=token_count,
                cumulative_events=self._event_counter,
                cumulative_ms=round(self._cumulative_ms, 4),
                metadata=metadata or {},
            )
            self._trace_buf.append(asdict(ev))
            if len(self._trace_buf) >= self._flush_every:
                self._flush()

        return ev

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    def trace_reconstruction(self, trigger: str, layer_indices: List[int]):
        return _ReconstructionCtx(self, trigger, layer_indices)

    # ------------------------------------------------------------------
    # Metrics queries
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._session_start
            rate = self._event_counter / max(elapsed, 1e-6)
            return {
                "total_events": self._event_counter,
                "total_reconstruction_ms": round(self._cumulative_ms, 4),
                "avg_duration_ms": round(self._cumulative_ms / max(self._event_counter, 1), 4),
                "rate_per_sec": round(rate, 4),
                "elapsed_sec": round(elapsed, 2),
                "trigger_breakdown": dict(self._trigger_counts),
                "trigger_ms_breakdown": {k: round(v, 4) for k, v in self._trigger_ms.items()},
            }

    def get_hot_layers(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return layers most frequently involved in dense reconstruction."""
        with self._lock:
            sorted_layers = sorted(
                self._layer_exposures.items(), key=lambda x: x[1], reverse=True
            )
            return [{"layer": li, "reconstruction_events": cnt}
                    for li, cnt in sorted_layers[:top_n]]

    def get_pressure_score(self) -> float:
        """
        Dense reconstruction pressure score in ms/sec.
        Lower is better; 0 means no dense reconstruction observed.
        """
        with self._lock:
            elapsed = max(time.time() - self._session_start, 1e-6)
            return self._cumulative_ms / elapsed

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        if not self._trace_buf:
            return
        with open(self.trace_path, "a") as f:
            for ev in self._trace_buf:
                f.write(json.dumps(ev) + "\n")
        self._trace_buf.clear()


class _ReconstructionCtx:
    def __init__(self, monitor: DenseReconstructionTraceMonitor,
                 trigger: str, layer_indices: List[int]):
        self._monitor = monitor
        self._trigger = trigger
        self._layers = layer_indices
        self._token_start = 0
        self._token_end = 0
        self._metadata: Dict = {}
        self._t0: float = 0.0

    def set_token_range(self, start: int, end: int) -> None:
        self._token_start = start
        self._token_end = end

    def set_metadata(self, **kwargs) -> None:
        self._metadata.update(kwargs)

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._t0) * 1000.0
        self._monitor.record_reconstruction(
            trigger=self._trigger,
            layer_indices=self._layers,
            token_start=self._token_start,
            token_end=self._token_end,
            duration_ms=duration_ms,
            metadata=self._metadata,
        )
        return False
