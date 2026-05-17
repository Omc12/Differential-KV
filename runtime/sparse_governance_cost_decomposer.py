"""
PRD Phase 41.0: Sparse Governance Cost Decomposer.
Isolates the TRUE per-token and per-request performance cost of sparse governance
subsystems. We are discovering whether governance overhead dominates savings.

Decomposes:
- semantic zoning cost
- repair system cost
- predictive scheduling cost
- equilibrium control cost
- semantic continuity monitoring cost
- telemetry generation cost
- trace persistence cost
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from contextlib import contextmanager


class SparseGovernanceCostDecomposer:
    """
    PRD Phase 41.0: Decomposes sparse governance overhead into discrete cost buckets.
    Each governance subsystem is timed individually to reveal the true overhead breakdown.
    """

    # Governance subsystem names — must match context manager names
    SUBSYSTEMS = [
        "semantic_zoning",
        "repair_system",
        "predictive_scheduling",
        "equilibrium_control",
        "continuity_monitoring",
        "telemetry_generation",
        "trace_persistence",
    ]

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("PRD_GovDecomposer")

        # Rolling windows per subsystem (last 120 samples)
        self._subsystem_latencies: Dict[str, deque] = {
            s: deque(maxlen=120) for s in self.SUBSYSTEMS
        }

        # Per-request accumulation
        self._request_costs: Dict[str, Dict[str, float]] = {}
        self._request_token_counts: Dict[str, int] = {}

        # Aggregate counters
        self._total_governance_sec: float = 0.0
        self._total_compute_sec: float = 0.0  # pure transformer time (reported externally)
        self._completed_requests: int = 0
        self._completed_tokens: int = 0

        self._trace_path = self.trace_dir / "governance_cost_trace.jsonl"
        self._logger.info(f"SparseGovernanceCostDecomposer initialized → {self.trace_dir}")

    # -----------------------------------------------------------------------
    # Context-manager timers — one per subsystem
    # -----------------------------------------------------------------------

    @contextmanager
    def time_semantic_zoning(self, request_id: str = "global"):
        yield from self._timer_ctx("semantic_zoning", request_id)

    @contextmanager
    def time_repair_system(self, request_id: str = "global"):
        yield from self._timer_ctx("repair_system", request_id)

    @contextmanager
    def time_predictive_scheduling(self, request_id: str = "global"):
        yield from self._timer_ctx("predictive_scheduling", request_id)

    @contextmanager
    def time_equilibrium_control(self, request_id: str = "global"):
        yield from self._timer_ctx("equilibrium_control", request_id)

    @contextmanager
    def time_continuity_monitoring(self, request_id: str = "global"):
        yield from self._timer_ctx("continuity_monitoring", request_id)

    @contextmanager
    def time_telemetry_generation(self, request_id: str = "global"):
        yield from self._timer_ctx("telemetry_generation", request_id)

    @contextmanager
    def time_trace_persistence(self, request_id: str = "global"):
        yield from self._timer_ctx("trace_persistence", request_id)

    def _timer_ctx(self, subsystem: str, request_id: str):
        """Shared implementation for all subsystem timers."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._subsystem_latencies[subsystem].append(elapsed)
                self._total_governance_sec += elapsed
                if request_id in self._request_costs:
                    self._request_costs[request_id][subsystem] = (
                        self._request_costs[request_id].get(subsystem, 0.0) + elapsed
                    )

    # -----------------------------------------------------------------------
    # Request lifecycle
    # -----------------------------------------------------------------------

    def request_started(self, request_id: str):
        with self._lock:
            self._request_costs[request_id] = {s: 0.0 for s in self.SUBSYSTEMS}
            self._request_token_counts[request_id] = 0

    def token_generated(self, request_id: str):
        with self._lock:
            if request_id in self._request_token_counts:
                self._request_token_counts[request_id] += 1

    def request_completed(
        self,
        request_id: str,
        total_request_sec: float,
        transformer_compute_sec: float,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if request_id not in self._request_costs:
                return None

            costs = self._request_costs.pop(request_id)
            token_count = max(self._request_token_counts.pop(request_id, 1), 1)
            total_gov = sum(costs.values())

            self._completed_requests += 1
            self._completed_tokens += token_count
            self._total_compute_sec += transformer_compute_sec

        per_token_costs = {
            f"{s}_per_token_us": round(costs[s] / token_count * 1_000_000, 1)
            for s in self.SUBSYSTEMS
        }
        pct_costs = {
            f"{s}_pct": round(costs[s] / total_request_sec * 100, 2)
            if total_request_sec > 0 else 0.0
            for s in self.SUBSYSTEMS
        }

        record = {
            "timestamp": time.time(),
            "request_id": request_id,
            "tokens": token_count,
            "total_request_sec": round(total_request_sec, 4),
            "transformer_compute_sec": round(transformer_compute_sec, 4),
            "total_governance_sec": round(total_gov, 4),
            "governance_overhead_pct": round(total_gov / total_request_sec * 100, 2)
            if total_request_sec > 0 else 0.0,
            "transformer_pct": round(transformer_compute_sec / total_request_sec * 100, 2)
            if total_request_sec > 0 else 0.0,
            **per_token_costs,
            **pct_costs,
        }
        self._persist(record)
        return record

    # -----------------------------------------------------------------------
    # Live reporting
    # -----------------------------------------------------------------------

    def get_live_summary(self) -> Dict[str, Any]:
        def safe_avg(dq: deque) -> float:
            return round(sum(dq) / len(dq) * 1000, 3) if dq else 0.0  # ms

        return {
            s: safe_avg(self._subsystem_latencies[s]) for s in self.SUBSYSTEMS
        }

    def get_total_governance_ratio(self) -> float:
        """Governance time as fraction of total compute time seen so far."""
        total = self._total_governance_sec + self._total_compute_sec
        if total <= 0:
            return 0.0
        return round(self._total_governance_sec / total, 4)

    def format_live_line(self) -> str:
        s = self.get_live_summary()
        ratio = self.get_total_governance_ratio()
        parts = [f"[GOV_DECOMP] gov_ratio={ratio:.1%}"]
        for sub in self.SUBSYSTEMS:
            parts.append(f"{sub}={s[sub]:.2f}ms")
        return " ".join(parts)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _persist(self, record: Dict[str, Any]):
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Governance trace persistence error: {e}")
