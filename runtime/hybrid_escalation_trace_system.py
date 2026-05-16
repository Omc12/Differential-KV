"""
STAGE 2 - SAG: Hybrid Escalation Trace System
Phase 39.0 - Sparse Attention Governance

Persists EVERY hybrid escalation event with full execution context.
This is the primary forensic trace for SAG analysis.

Each record captures:
  - triggering step and layer
  - confidence score at escalation moment
  - fallback reason (from window controller / suppression layer)
  - sparse window state at escalation
  - escalation duration (how long hybrid persisted before returning to sparse)
  - whether suppression was attempted and outcome
  - token range and sequence position

This trace is append-only and immutable during a run.
"""

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from runtime.sparse_confidence_estimator import _JsonlWriter


class HybridEscalationTraceSystem:
    """
    STAGE 2 SAG: Hybrid Escalation Trace System.

    Record every transition into hybrid/dense mode with full context.
    Also track escalation duration (steps until return to sparse).

    Usage:
        # When entering hybrid/dense:
        eid = tracer.record_escalation_start(
            step, layer_idx, from_mode="sparse", to_mode="hybrid",
            confidence=..., gate_score=..., reason=...,
            window_size=..., seq_len=..., token_offset=...,
            suppression_attempted=True,
        )

        # When returning to sparse:
        tracer.record_escalation_end(eid, step, return_mode="sparse")
    """

    def __init__(
        self,
        trace_path: str = "traces/stage2/phase_39_0_sag/hybrid_escalation_trace.jsonl",
    ):
        self._writer = _JsonlWriter(trace_path)
        self._lock = threading.Lock()

        # Open escalations per layer (layer_idx -> escalation record)
        self._open: Dict[int, Dict] = {}
        self._event_counter = 0

        # Aggregate stats
        self._total_escalations   = 0
        self._total_resolved      = 0
        self._escalation_dur_sum  = 0   # steps
        self._layer_escalations:  Dict[int, int] = defaultdict(int)
        self._reason_counts:      Dict[str, int] = defaultdict(int)
        self._session_start = time.time()

    # ------------------------------------------------------------------

    def record_escalation_start(
        self,
        step: int,
        layer_idx: int,
        from_mode: str,
        to_mode: str,
        confidence: float,
        gate_score: float,
        reason: str,
        window_size: int,
        seq_len: int,
        token_offset: int,
        suppression_attempted: bool,
    ) -> int:
        """
        Record the start of a hybrid/dense escalation.

        Returns:
            event_id (pass to record_escalation_end when resolved)
        """
        ts = time.time()

        with self._lock:
            self._event_counter += 1
            eid = self._event_counter
            self._total_escalations += 1
            self._layer_escalations[layer_idx] += 1
            self._reason_counts[reason] += 1

            record = {
                "event_id": eid,
                "ts_start": ts,
                "step_start": step,
                "layer_idx": layer_idx,
                "from_mode": from_mode,
                "to_mode": to_mode,
                "confidence": round(confidence, 4),
                "gate_score": round(gate_score, 4),
                "reason": reason,
                "window_size": window_size,
                "seq_len": seq_len,
                "token_offset": token_offset,
                "suppression_attempted": suppression_attempted,
                "event_type": "escalation_start",
                "phase": "39.0-SAG",
            }
            self._open[layer_idx] = record

        self._writer.write(record)
        return eid

    def record_escalation_end(
        self,
        event_id: int,
        layer_idx: int,
        step_end: int,
        return_mode: str,
    ) -> Optional[int]:
        """
        Record resolution of an escalation.

        Returns:
            duration_steps (None if no open escalation for this layer)
        """
        ts = time.time()

        with self._lock:
            if layer_idx not in self._open:
                return None

            start_rec = self._open.pop(layer_idx)
            duration_steps = step_end - start_rec["step_start"]
            self._escalation_dur_sum += duration_steps
            self._total_resolved += 1

        record = {
            "event_id": event_id,
            "ts_end": ts,
            "step_end": step_end,
            "layer_idx": layer_idx,
            "return_mode": return_mode,
            "duration_steps": duration_steps,
            "event_type": "escalation_end",
            "phase": "39.0-SAG",
        }
        self._writer.write(record)
        return duration_steps

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            resolved = max(self._total_resolved, 1)
            return {
                "total_escalations":    self._total_escalations,
                "total_resolved":       self._total_resolved,
                "currently_open":       len(self._open),
                "mean_duration_steps":  round(self._escalation_dur_sum / resolved, 2),
                "top_reason":           max(self._reason_counts, key=self._reason_counts.get)
                                        if self._reason_counts else "none",
                "reason_distribution":  dict(self._reason_counts),
                "elapsed_sec":          round(time.time() - self._session_start, 2),
            }

    def get_layer_escalation_counts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"layer": l, "escalations": c}
                for l, c in sorted(self._layer_escalations.items())
            ]

    def flush_and_close(self) -> None:
        # Close any still-open escalations
        ts = time.time()
        with self._lock:
            for layer_idx, rec in self._open.items():
                self._writer.write({
                    "event_id": rec["event_id"],
                    "ts_end": ts,
                    "layer_idx": layer_idx,
                    "event_type": "escalation_unclosed",
                    "note": "run ended while escalation was open",
                    "phase": "39.0-SAG",
                })
        self._writer.flush_and_close()
