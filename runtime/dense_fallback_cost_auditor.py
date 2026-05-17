"""
PRD Phase 41.0: Dense Fallback Cost Auditor.
Measures the TRUE cost of dense recovery passes, semantic repairs,
hybrid routing, fallback escalation, and anchor reinforcement.

Dense fallback may be erasing all sparse gains.

Tracks:
- fallback frequency (events/sec)
- fallback duration (ms per event)
- cumulative dense overhead (% of total compute)
- recovered vs unrecovered cost
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
from contextlib import contextmanager


class DenseFallbackCostAuditor:
    """
    PRD Phase 41.0: Audits the real cost of dense fallback operations.
    Every fallback event is timed and persisted for post-hoc decomposition.
    """

    FALLBACK_TYPES = [
        "dense_recovery_pass",
        "semantic_repair",
        "hybrid_routing",
        "fallback_escalation",
        "anchor_reinforcement",
    ]

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("PRD_FallbackAuditor")

        # Per-type tracking
        self._fallback_counts: Dict[str, int] = {t: 0 for t in self.FALLBACK_TYPES}
        self._fallback_durations: Dict[str, deque] = {
            t: deque(maxlen=200) for t in self.FALLBACK_TYPES
        }
        self._fallback_recovered: Dict[str, int] = {t: 0 for t in self.FALLBACK_TYPES}
        self._fallback_unrecovered: Dict[str, int] = {t: 0 for t in self.FALLBACK_TYPES}

        # Global tracking
        self._total_fallback_sec: float = 0.0
        self._total_compute_sec: float = 0.0  # reported externally
        self._session_start_ts: float = time.time()
        self._recent_events: deque = deque(maxlen=500)

        self._trace_path = self.trace_dir / "dense_fallback_trace.jsonl"
        self._logger.info(f"DenseFallbackCostAuditor initialized → {self.trace_dir}")

    # -----------------------------------------------------------------------
    # Context-manager based timing per fallback type
    # -----------------------------------------------------------------------

    @contextmanager
    def time_dense_recovery_pass(self, request_id: str = "global", recovered: bool = True):
        yield from self._fallback_ctx("dense_recovery_pass", request_id, recovered)

    @contextmanager
    def time_semantic_repair(self, request_id: str = "global", recovered: bool = True):
        yield from self._fallback_ctx("semantic_repair", request_id, recovered)

    @contextmanager
    def time_hybrid_routing(self, request_id: str = "global", recovered: bool = True):
        yield from self._fallback_ctx("hybrid_routing", request_id, recovered)

    @contextmanager
    def time_fallback_escalation(self, request_id: str = "global", recovered: bool = True):
        yield from self._fallback_ctx("fallback_escalation", request_id, recovered)

    @contextmanager
    def time_anchor_reinforcement(self, request_id: str = "global", recovered: bool = True):
        yield from self._fallback_ctx("anchor_reinforcement", request_id, recovered)

    def _fallback_ctx(self, fallback_type: str, request_id: str, recovered: bool):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record_fallback(fallback_type, request_id, elapsed, recovered)

    # -----------------------------------------------------------------------
    # Direct recording
    # -----------------------------------------------------------------------

    def record_fallback(
        self,
        fallback_type: str,
        duration_sec: float,
        request_id: str = "global",
        recovered: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a fallback event from external timing."""
        self._record_fallback(fallback_type, request_id, duration_sec, recovered, metadata)

    def report_compute_time(self, compute_sec: float):
        """Report pure transformer compute time (for overhead ratio computation)."""
        with self._lock:
            self._total_compute_sec += compute_sec

    # -----------------------------------------------------------------------
    # Aggregated view
    # -----------------------------------------------------------------------

    def get_live_summary(self) -> Dict[str, Any]:
        with self._lock:
            elapsed_session = time.time() - self._session_start_ts
            total_events = sum(self._fallback_counts.values())
            fallback_rate = round(total_events / elapsed_session, 3) if elapsed_session > 0 else 0.0

            total_recovered = sum(self._fallback_recovered.values())
            total_unrecovered = sum(self._fallback_unrecovered.values())

            total_fallback = self._total_fallback_sec
            total_compute = self._total_compute_sec
            overhead_pct = round(
                total_fallback / (total_fallback + total_compute) * 100, 2
            ) if (total_fallback + total_compute) > 0 else 0.0

            per_type = {}
            for t in self.FALLBACK_TYPES:
                dq = self._fallback_durations[t]
                avg_ms = round(sum(dq) / len(dq) * 1000, 2) if dq else 0.0
                per_type[t] = {
                    "count": self._fallback_counts[t],
                    "avg_duration_ms": avg_ms,
                    "recovered": self._fallback_recovered[t],
                    "unrecovered": self._fallback_unrecovered[t],
                }

        return {
            "total_fallback_events": total_events,
            "fallback_rate_per_sec": fallback_rate,
            "total_fallback_sec": round(total_fallback, 4),
            "cumulative_fallback_overhead_pct": overhead_pct,
            "total_recovered": total_recovered,
            "total_unrecovered": total_unrecovered,
            "recovery_rate": round(total_recovered / max(total_events, 1), 3),
            "per_type": per_type,
        }

    def format_live_line(self) -> str:
        s = self.get_live_summary()
        return (
            f"[FALLBACK] events={s['total_fallback_events']} "
            f"rate={s['fallback_rate_per_sec']:.2f}/s "
            f"overhead={s['cumulative_fallback_overhead_pct']:.1f}% "
            f"recovery_rate={s['recovery_rate']:.1%} "
            f"unrecovered={s['total_unrecovered']}"
        )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _record_fallback(
        self,
        fallback_type: str,
        request_id: str,
        duration_sec: float,
        recovered: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        with self._lock:
            if fallback_type not in self._fallback_counts:
                # Unknown type — create entry
                self._fallback_counts[fallback_type] = 0
                self._fallback_durations[fallback_type] = deque(maxlen=200)
                self._fallback_recovered[fallback_type] = 0
                self._fallback_unrecovered[fallback_type] = 0

            self._fallback_counts[fallback_type] += 1
            self._fallback_durations[fallback_type].append(duration_sec)
            self._total_fallback_sec += duration_sec

            if recovered:
                self._fallback_recovered[fallback_type] += 1
            else:
                self._fallback_unrecovered[fallback_type] += 1

            record = {
                "timestamp": time.time(),
                "fallback_type": fallback_type,
                "request_id": request_id,
                "duration_sec": round(duration_sec, 5),
                "duration_ms": round(duration_sec * 1000, 3),
                "recovered": recovered,
                "cumulative_total_fallback_sec": round(self._total_fallback_sec, 4),
            }
            if metadata:
                record.update(metadata)
            self._recent_events.append(record)

        self._persist(record)

    def _persist(self, record: Dict[str, Any]):
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Fallback trace persistence error: {e}")
