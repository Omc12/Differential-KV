"""
STAGE 2 - SAG: Adaptive Hybrid Suppression Layer
Phase 39.0 - Sparse Attention Governance

Reduces unnecessary hybrid escalation by intercepting escalation decisions
and suppressing them when confidence is sufficiently high.

Design:
  - Confidence-aware: suppression threshold adapts to real confidence signal
  - Layer-aware: earlier layers tolerate more suppression; later = conservative
  - Persistence bias: sparse mode holds for N steps before reconsidering
  - Safety: consecutive fail limit auto-disengages suppression per layer
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

from runtime.sparse_confidence_estimator import _JsonlWriter


class AdaptiveHybridSuppressionLayer:
    """
    STAGE 2 SAG: Adaptive Hybrid Suppression Layer.

    Evaluate each proposed attention mode and suppress hybrid escalations
    when real confidence evidence supports staying sparse.

    Usage:
        decision = suppression_layer.evaluate(
            layer_idx=4, proposed_mode="hybrid",
            confidence=0.81, gate_score=0.76,
            fallback_rate=0.12, step=step,
        )
        actual_mode = decision.resolved_mode
    """

    _EARLY_SUPPRESS_THRESHOLD  = 0.65
    _MIDDLE_SUPPRESS_THRESHOLD = 0.72
    _LATE_SUPPRESS_THRESHOLD   = 0.80
    _CONSECUTIVE_FAIL_LIMIT    = 3
    _SPARSE_PERSISTENCE_STEPS  = 2

    def __init__(
        self,
        num_layers: int = 28,
        trace_path: str = "traces/stage2/phase_39_0_sag/hybrid_escalation_trace.jsonl",
    ):
        self.num_layers = num_layers
        self._writer = _JsonlWriter(trace_path)
        self._lock = threading.Lock()

        self._consecutive_fails:  Dict[int, int]  = defaultdict(int)
        self._layer_disengaged:   Dict[int, bool] = defaultdict(bool)
        self._sparse_persist_rem: Dict[int, int]  = defaultdict(int)

        self._prevented_escalations  = 0
        self._failed_suppressions    = 0
        self._successful_persistence = 0
        self._total_evaluated        = 0
        self._layer_prevented: Dict[int, int] = defaultdict(int)
        self._layer_failed:    Dict[int, int] = defaultdict(int)
        self._session_start = time.time()

    # -------------------------------------------------------------------

    class Decision:
        __slots__ = ("resolved_mode", "suppressed", "suppression_reason",
                     "original_mode", "confidence_at_decision")

        def __init__(self, resolved_mode, suppressed, suppression_reason,
                     original_mode, confidence):
            self.resolved_mode = resolved_mode
            self.suppressed = suppressed
            self.suppression_reason = suppression_reason
            self.original_mode = original_mode
            self.confidence_at_decision = confidence

    def evaluate(
        self,
        layer_idx: int,
        proposed_mode: str,
        confidence: float,
        gate_score: float,
        fallback_rate: float,
        step: int,
    ) -> "AdaptiveHybridSuppressionLayer.Decision":
        ts = time.time()

        with self._lock:
            self._total_evaluated += 1
            suppressed = False
            suppression_reason = None
            resolved_mode = proposed_mode

            if proposed_mode == "hybrid" and not self._layer_disengaged[layer_idx]:
                threshold = self._suppression_threshold(layer_idx)

                if self._sparse_persist_rem[layer_idx] > 0:
                    resolved_mode = "sparse"
                    suppressed = True
                    suppression_reason = f"persistence={self._sparse_persist_rem[layer_idx]}"
                    self._sparse_persist_rem[layer_idx] -= 1
                    self._prevented_escalations += 1
                    self._layer_prevented[layer_idx] += 1
                elif confidence >= threshold:
                    resolved_mode = "sparse"
                    suppressed = True
                    suppression_reason = (
                        f"conf={confidence:.3f}>={threshold:.3f} gate={gate_score:.3f}"
                    )
                    self._sparse_persist_rem[layer_idx] = self._SPARSE_PERSISTENCE_STEPS
                    self._prevented_escalations += 1
                    self._layer_prevented[layer_idx] += 1

        self._writer.write({
            "ts": ts, "step": step, "layer_idx": layer_idx,
            "proposed_mode": proposed_mode, "resolved_mode": resolved_mode,
            "suppressed": suppressed, "suppression_reason": suppression_reason,
            "confidence": round(confidence, 4), "gate_score": round(gate_score, 4),
            "fallback_rate": round(fallback_rate, 4), "phase": "39.0-SAG",
        })
        return self.Decision(resolved_mode, suppressed, suppression_reason,
                             proposed_mode, confidence)

    def record_outcome(self, layer_idx: int, suppressed: bool,
                       actual_fallback_occurred: bool) -> None:
        with self._lock:
            if suppressed:
                if actual_fallback_occurred:
                    self._failed_suppressions += 1
                    self._layer_failed[layer_idx] += 1
                    self._consecutive_fails[layer_idx] += 1
                    if self._consecutive_fails[layer_idx] >= self._CONSECUTIVE_FAIL_LIMIT:
                        self._layer_disengaged[layer_idx] = True
                else:
                    self._successful_persistence += 1
                    self._consecutive_fails[layer_idx] = 0

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_evaluated, 1)
            return {
                "total_evaluated":        self._total_evaluated,
                "prevented_escalations":  self._prevented_escalations,
                "failed_suppressions":    self._failed_suppressions,
                "successful_persistence": self._successful_persistence,
                "suppression_rate":       round(self._prevented_escalations / total, 4),
                "failure_rate": round(
                    self._failed_suppressions / max(self._prevented_escalations, 1), 4),
                "disengaged_layers":
                    [l for l, d in self._layer_disengaged.items() if d],
                "elapsed_sec": round(time.time() - self._session_start, 2),
            }

    def get_layer_stats(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"layer": li, "prevented": self._layer_prevented[li],
                 "failed": self._layer_failed[li],
                 "disengaged": self._layer_disengaged[li]}
                for li in sorted(set(self._layer_prevented) | set(self._layer_failed))
            ]

    def flush_and_close(self) -> None:
        self._writer.flush_and_close()

    def _suppression_threshold(self, layer_idx: int) -> float:
        depth = layer_idx / max(self.num_layers - 1, 1)
        if depth < 0.33:
            return self._EARLY_SUPPRESS_THRESHOLD
        elif depth < 0.67:
            t = (depth - 0.33) / 0.34
            return self._EARLY_SUPPRESS_THRESHOLD + t * (
                self._MIDDLE_SUPPRESS_THRESHOLD - self._EARLY_SUPPRESS_THRESHOLD)
        else:
            t = (depth - 0.67) / 0.33
            return self._MIDDLE_SUPPRESS_THRESHOLD + t * (
                self._LATE_SUPPRESS_THRESHOLD - self._MIDDLE_SUPPRESS_THRESHOLD)
