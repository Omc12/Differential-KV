"""
STAGE 2 - SAG: Sparse Confidence Estimator
Phase 39.0 - Sparse Attention Governance

Estimates confidence that sparse attention will preserve semantic integrity
for a given layer/step, derived exclusively from real runtime signals.

Signals used (all caller-provided from actual execution):
  - gate_score_history:    recent gate scores from SparseAttentionWindowController
  - window_hit_rate:       fraction of attended tokens that fell within sparse window
  - token_entropy:         derived from gate score distribution spread
  - fallback_rate:         recent fallback frequency from path auditor
  - step_delta:            confidence change between consecutive steps

Confidence range: [0.0, 1.0]
  1.0 = high confidence that sparse will preserve semantics
  0.0 = sparse path likely to miss critical context

All values MUST come from real execution measurements.
No synthetic estimation or fixed lookup tables.
"""

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# JSONL writer (shared pattern across SAG components)
# ---------------------------------------------------------------------------
import json
import os


class _JsonlWriter:
    def __init__(self, path: str, buffer_size: int = 128):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path
        self._buffer: List[Dict] = []
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, record: Dict) -> None:
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._buffer_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        for r in self._buffer:
            self._fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        self._buffer.clear()
        self._fh.flush()

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._fh.close()


# ---------------------------------------------------------------------------
# SparseConfidenceEstimator
# ---------------------------------------------------------------------------

class SparseConfidenceEstimator:
    """
    STAGE 2 SAG: Sparse Confidence Estimator.

    Produces a single confidence score [0, 1] per attention call, based on
    real runtime signals. Higher confidence = safer to stay sparse.

    Caller responsibilities:
      Record each attention result via record_attention_outcome(), passing:
        - gate_score:     float [0,1] from SparseAttentionWindowController
        - window_hit_rate: float [0,1] — fraction of tokens within the window
                          that the model actually attended to (from sparse kernel)
        - fallback_rate:  recent dense fallback rate from SparseAttentionPathAuditor
        - seq_len:        current sequence length
        - layer_idx:      transformer layer index

    The estimator maintains sliding windows per layer to track stability.
    """

    _WINDOW = 16          # sliding window depth per layer for stability tracking
    _MIN_SAMPLES = 4      # minimum samples before confidence is meaningful
    _ENTROPY_SCALE = 4.0  # scaling factor for gate score spread to entropy estimate

    def __init__(
        self,
        num_layers: int = 28,
        trace_path: str = "traces/stage2/phase_39_0_sag/sparse_confidence_trace.jsonl",
    ):
        self.num_layers = num_layers
        self._writer = _JsonlWriter(trace_path)
        self._lock = threading.Lock()

        # Per-layer sliding windows of real signals
        self._gate_history:    Dict[int, Deque[float]] = {}
        self._hit_history:     Dict[int, Deque[float]] = {}
        self._fallback_history:Dict[int, Deque[float]] = {}

        # Running confidence per layer
        self._layer_confidence: Dict[int, float] = {}

        # Global counters
        self._total_events = 0
        self._high_confidence_events = 0   # >= 0.7
        self._low_confidence_events  = 0   # < 0.4
        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def record_attention_outcome(
        self,
        step: int,
        layer_idx: int,
        gate_score: float,
        window_hit_rate: float,
        fallback_rate: float,
        seq_len: int,
        mode: str,
    ) -> float:
        """
        Record one real attention outcome and return the updated confidence.

        Args:
            step:           global decode step counter
            layer_idx:      transformer layer (0-based)
            gate_score:     gate score from window controller [0,1]
            window_hit_rate: fraction of window tokens actually attended [0,1]
            fallback_rate:  recent fallback rate from path auditor [0,1]
            seq_len:        current sequence length
            mode:           actual mode used ("sparse"/"hybrid"/"dense")

        Returns:
            confidence: float [0, 1]
        """
        ts = time.time()

        with self._lock:
            # Initialise deques on first use
            if layer_idx not in self._gate_history:
                self._gate_history[layer_idx]     = deque(maxlen=self._WINDOW)
                self._hit_history[layer_idx]      = deque(maxlen=self._WINDOW)
                self._fallback_history[layer_idx] = deque(maxlen=self._WINDOW)

            # Append real signals
            self._gate_history[layer_idx].append(gate_score)
            self._hit_history[layer_idx].append(window_hit_rate)
            self._fallback_history[layer_idx].append(fallback_rate)

            # Compute confidence from real signals
            confidence = self._compute_confidence(layer_idx, seq_len)
            self._layer_confidence[layer_idx] = confidence

            self._total_events += 1
            if confidence >= 0.7:
                self._high_confidence_events += 1
            elif confidence < 0.4:
                self._low_confidence_events += 1

        # Write trace record
        record = {
            "ts": ts,
            "step": step,
            "layer_idx": layer_idx,
            "gate_score": round(gate_score, 4),
            "window_hit_rate": round(window_hit_rate, 4),
            "fallback_rate": round(fallback_rate, 4),
            "seq_len": seq_len,
            "mode": mode,
            "confidence": round(confidence, 4),
            "phase": "39.0-SAG",
        }
        self._writer.write(record)
        return confidence

    def get_layer_confidence(self, layer_idx: int) -> float:
        """Return the most recent computed confidence for a given layer."""
        with self._lock:
            return self._layer_confidence.get(layer_idx, 0.5)

    def get_global_confidence(self) -> float:
        """Mean confidence across all layers that have been observed."""
        with self._lock:
            if not self._layer_confidence:
                return 0.5
            return round(sum(self._layer_confidence.values()) / len(self._layer_confidence), 4)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_events, 1)
            return {
                "total_events": self._total_events,
                "high_confidence_rate": round(self._high_confidence_events / total, 4),
                "low_confidence_rate":  round(self._low_confidence_events  / total, 4),
                "global_confidence":    self.get_global_confidence(),
                "elapsed_sec": round(time.time() - self._session_start, 2),
            }

    def flush_and_close(self) -> None:
        self._writer.flush_and_close()

    # ------------------------------------------------------------------
    # Internal: confidence from real signals
    # ------------------------------------------------------------------

    def _compute_confidence(self, layer_idx: int, seq_len: int) -> float:
        """
        Derive confidence from real sliding-window signal history.

        Components:
          1. gate_stability:   low variance in gate scores -> stable sparse decision
          2. hit_quality:      high window hit rate -> window is capturing real attention
          3. fallback_penalty: high recent fallback rate -> confidence reduced
          4. seq_pressure:     very long sequences reduce confidence slightly

        All components derived from real measurements, weighted equally.
        """
        gates     = list(self._gate_history[layer_idx])
        hits      = list(self._hit_history[layer_idx])
        fallbacks = list(self._fallback_history[layer_idx])

        if len(gates) < self._MIN_SAMPLES:
            # Not enough history yet — use raw gate score as proxy
            return min(max(gates[-1], 0.0), 1.0) if gates else 0.5

        # 1. Gate stability: 1.0 when gate scores are consistent
        gate_mean = sum(gates) / len(gates)
        gate_var  = sum((g - gate_mean) ** 2 for g in gates) / len(gates)
        gate_std  = math.sqrt(gate_var)
        gate_stability = max(0.0, 1.0 - gate_std * self._ENTROPY_SCALE)

        # 2. Hit quality: direct signal from window coverage
        hit_mean = sum(hits) / len(hits)
        # Penalise if hit rate is declining over the window
        if len(hits) >= 4:
            early = sum(list(hits)[:len(hits)//2]) / (len(hits)//2)
            late  = sum(list(hits)[len(hits)//2:]) / (len(hits) - len(hits)//2)
            drift = max(0.0, early - late)
        else:
            drift = 0.0
        hit_quality = max(0.0, hit_mean - drift)

        # 3. Fallback penalty: recent fallback rate erodes confidence
        fallback_mean = sum(fallbacks) / len(fallbacks)
        fallback_penalty = fallback_mean  # [0, 1] — subtracted below

        # 4. Sequence pressure: gently penalise very long sequences
        #    (they are harder to cover with sparse windows)
        seq_pressure = min(0.2, seq_len / 32768.0)

        # Combine
        confidence = (
            0.35 * gate_stability +
            0.40 * hit_quality +
            0.25 * (1.0 - fallback_penalty)
        ) - seq_pressure * 0.1

        return round(min(max(confidence, 0.0), 1.0), 4)
