"""
PRD Phase 41.0: Runtime Control-Plane Weight Analyzer.
Quantifies how much runtime effort is spent on CONTROL vs COMPUTE.

Computes:
- governance compute %
- telemetry compute %
- orchestration compute %
- actual transformer compute %
- synchronization overhead %

This is one of the most critical measurements in the project.
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from contextlib import contextmanager


class RuntimeControlPlaneWeightAnalyzer:
    """
    PRD Phase 41.0: Decomposes runtime time budget into control-plane vs. data-plane.

    A runtime that spends 40% of its time on governance, telemetry, and
    orchestration cannot achieve meaningful sparse acceleration regardless
    of how efficient the sparse kernel is.
    """

    # All trackable compute categories
    CATEGORIES = [
        "transformer_compute",   # Actual forward pass (data-plane)
        "governance",            # Semantic zoning, equilibrium, continuity
        "telemetry",             # Metrics collection and emission
        "orchestration",         # Scheduling, batching, queue management
        "synchronization",       # CUDA sync, thread sync, async awaits
        "trace_io",              # Trace file I/O
        "recovery",              # Dense fallback, repair passes
        "routing",               # Sparse/dense routing decisions
        "other",                 # Uncategorized overhead
    ]

    def __init__(self, trace_dir: Path, window_sec: float = 30.0):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("PRD_ControlPlaneAnalyzer")
        self._window_sec = window_sec

        # Time-stamped samples per category for rolling window
        self._category_samples: Dict[str, deque] = {
            c: deque(maxlen=2000) for c in self.CATEGORIES
        }

        # Cumulative totals (session-wide)
        self._cumulative_sec: Dict[str, float] = {c: 0.0 for c in self.CATEGORIES}
        self._session_start = time.time()

        # Per-request accumulation
        self._request_budgets: Dict[str, Dict[str, float]] = {}

        self._trace_path = self.trace_dir / "control_plane_trace.jsonl"
        self._snapshot_interval = 5.0
        self._last_snapshot = time.time()
        self._logger.info(f"RuntimeControlPlaneWeightAnalyzer initialized → {self.trace_dir}")

    # -----------------------------------------------------------------------
    # Context-manager timers — one per category
    # -----------------------------------------------------------------------

    @contextmanager
    def time_transformer(self, request_id: Optional[str] = None):
        yield from self._category_ctx("transformer_compute", request_id)

    @contextmanager
    def time_governance(self, request_id: Optional[str] = None):
        yield from self._category_ctx("governance", request_id)

    @contextmanager
    def time_telemetry(self, request_id: Optional[str] = None):
        yield from self._category_ctx("telemetry", request_id)

    @contextmanager
    def time_orchestration(self, request_id: Optional[str] = None):
        yield from self._category_ctx("orchestration", request_id)

    @contextmanager
    def time_synchronization(self, request_id: Optional[str] = None):
        yield from self._category_ctx("synchronization", request_id)

    @contextmanager
    def time_trace_io(self, request_id: Optional[str] = None):
        yield from self._category_ctx("trace_io", request_id)

    @contextmanager
    def time_recovery(self, request_id: Optional[str] = None):
        yield from self._category_ctx("recovery", request_id)

    @contextmanager
    def time_routing(self, request_id: Optional[str] = None):
        yield from self._category_ctx("routing", request_id)

    def _category_ctx(self, category: str, request_id: Optional[str]):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record(category, elapsed, request_id)

    # -----------------------------------------------------------------------
    # Direct recording
    # -----------------------------------------------------------------------

    def record(self, category: str, duration_sec: float, request_id: Optional[str] = None):
        """Record time spent in a given category directly."""
        self._record(category, duration_sec, request_id)

    def request_started(self, request_id: str):
        with self._lock:
            self._request_budgets[request_id] = {c: 0.0 for c in self.CATEGORIES}

    def request_completed(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if request_id not in self._request_budgets:
                return None
            budget = self._request_budgets.pop(request_id)

        total = sum(budget.values())
        if total <= 0:
            return None

        pct = {f"{c}_pct": round(budget[c] / total * 100, 2) for c in self.CATEGORIES}
        control_plane = total - budget["transformer_compute"]
        control_pct = round(control_plane / total * 100, 2) if total > 0 else 0.0
        compute_pct = round(budget["transformer_compute"] / total * 100, 2) if total > 0 else 0.0

        record = {
            "timestamp": time.time(),
            "request_id": request_id,
            "total_wall_sec": round(total, 4),
            "control_plane_sec": round(control_plane, 4),
            "transformer_compute_sec": round(budget["transformer_compute"], 4),
            "control_plane_pct": control_pct,
            "transformer_compute_pct": compute_pct,
            **pct,
        }
        self._persist(record)
        return record

    # -----------------------------------------------------------------------
    # Snapshot emitter — call periodically to persist rolling summary
    # -----------------------------------------------------------------------

    def maybe_emit_snapshot(self):
        now = time.time()
        with self._lock:
            if now - self._last_snapshot < self._snapshot_interval:
                return
            self._last_snapshot = now
            snapshot = self._build_snapshot()
        self._persist(snapshot)

    # -----------------------------------------------------------------------
    # Live reporting
    # -----------------------------------------------------------------------

    def get_live_weights(self) -> Dict[str, float]:
        """
        Returns the fraction of runtime consumed by each category,
        computed over the rolling window.
        """
        with self._lock:
            window_start = time.time() - self._window_sec
            window_totals: Dict[str, float] = {c: 0.0 for c in self.CATEGORIES}

            for category, samples in self._category_samples.items():
                for ts, dur in samples:
                    if ts >= window_start:
                        window_totals[category] += dur

            grand_total = sum(window_totals.values())
            if grand_total <= 0:
                return {c: 0.0 for c in self.CATEGORIES}

            return {c: round(window_totals[c] / grand_total, 4) for c in self.CATEGORIES}

    def get_control_plane_ratio(self) -> float:
        weights = self.get_live_weights()
        transformer = weights.get("transformer_compute", 0.0)
        return round(1.0 - transformer, 4)

    def format_live_line(self) -> str:
        weights = self.get_live_weights()
        cp_ratio = self.get_control_plane_ratio()
        transformer_pct = round(weights.get("transformer_compute", 0) * 100, 1)
        gov_pct = round(weights.get("governance", 0) * 100, 1)
        telem_pct = round(weights.get("telemetry", 0) * 100, 1)
        orch_pct = round(weights.get("orchestration", 0) * 100, 1)
        sync_pct = round(weights.get("synchronization", 0) * 100, 1)
        rec_pct = round(weights.get("recovery", 0) * 100, 1)

        return (
            f"[CONTROL_PLANE] cp={cp_ratio:.1%} "
            f"transformer={transformer_pct:.1f}% "
            f"gov={gov_pct:.1f}% "
            f"telem={telem_pct:.1f}% "
            f"orch={orch_pct:.1f}% "
            f"sync={sync_pct:.1f}% "
            f"recovery={rec_pct:.1f}%"
        )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _record(self, category: str, elapsed: float, request_id: Optional[str]):
        if category not in self._cumulative_sec:
            category = "other"
        ts = time.time()
        with self._lock:
            self._category_samples[category].append((ts, elapsed))
            self._cumulative_sec[category] += elapsed
            if request_id and request_id in self._request_budgets:
                self._request_budgets[request_id][category] += elapsed

    def _build_snapshot(self) -> Dict[str, Any]:
        weights = self.get_live_weights()  # called within lock context — avoid deadlock
        # Get without lock since caller holds it
        session_elapsed = time.time() - self._session_start
        total_cum = sum(self._cumulative_sec.values())
        cp_cum = total_cum - self._cumulative_sec.get("transformer_compute", 0.0)

        pct_cum = {
            f"{c}_cumulative_pct": round(self._cumulative_sec[c] / total_cum * 100, 2)
            for c in self.CATEGORIES
        } if total_cum > 0 else {}

        return {
            "timestamp": time.time(),
            "event_type": "snapshot",
            "session_elapsed_sec": round(session_elapsed, 1),
            "cumulative_total_sec": round(total_cum, 3),
            "cumulative_control_plane_sec": round(cp_cum, 3),
            "cumulative_control_plane_pct": round(cp_cum / total_cum * 100, 2) if total_cum > 0 else 0.0,
            **pct_cum,
        }

    def _persist(self, record: Dict[str, Any]):
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Control plane trace persistence error: {e}")
