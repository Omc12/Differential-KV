"""
RCO-N Phase 41.1: Runtime Collapse Coordinator.

Collapses:
- redundant governance wakeups
- fragmented orchestration loops
- unnecessary synchronization cycles
- excessive telemetry polling

Governance becomes LIGHTWEIGHT CONTROL SIGNALS,
not a mini-runtime per token.
"""

import time
import threading
import logging
import asyncio
from collections import defaultdict, deque
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path
import json


class GovernanceWindow:
    """
    A batched governance execution window.
    Instead of firing governance checks per-token, we batch them
    into unified windows that fire every N tokens or T milliseconds.
    """
    __slots__ = ["window_id", "token_count", "session_ids", "open_ts", "signals"]

    def __init__(self, window_id: int):
        self.window_id = window_id
        self.token_count = 0
        self.session_ids: List[str] = []
        self.open_ts = time.perf_counter()
        self.signals: Dict[str, Any] = {}


class RuntimeCollapseCoordinator:
    """
    RCO-N Phase 41.1: Collapses per-token governance and orchestration overhead
    into unified, batched execution windows.

    Key design decisions:
    1. Governance fires every GOVERNANCE_WINDOW_TOKENS tokens (not per-token)
    2. Telemetry is aggregated and flushed every TELEMETRY_FLUSH_INTERVAL_MS
    3. Synchronization cycles are coalesced across sessions
    4. Orchestration wakeups are deduplicated using a work-coalescing queue
    """

    GOVERNANCE_WINDOW_TOKENS   = 8     # Fire governance every N tokens across all sessions
    TELEMETRY_FLUSH_INTERVAL_S = 0.5   # Flush telemetry every 500ms (not per-token)
    ORCHESTRATION_COALESCE_MS  = 5     # Coalesce orchestration wakeups within 5ms windows
    MAX_SYNC_COALESCE          = 32    # Max items to batch in one sync flush

    def __init__(self, trace_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._logger = logging.getLogger("RCO_CollapseCoordinator")

        # Governance batching
        self._current_window = GovernanceWindow(0)
        self._window_token_counter = 0
        self._governance_callbacks: List[Callable] = []
        self._governance_fires = 0
        self._governance_skips = 0  # tokens where governance was correctly skipped

        # Telemetry batching
        self._telemetry_buffer: Dict[str, List[Any]] = defaultdict(list)
        self._telemetry_flush_callbacks: List[Callable] = []
        self._last_telemetry_flush = time.perf_counter()
        self._telemetry_suppressed_count = 0  # Suppressed calls since last flush

        # Orchestration coalescing
        self._pending_orchestration: deque = deque()
        self._orchestration_callbacks: List[Callable] = []
        self._orchestration_coalesced = 0

        # Sync cycle tracking
        self._sync_coalesce_buffer: List[Any] = []
        self._sync_fires = 0
        self._sync_coalesced = 0

        # Session activity window
        self._active_sessions: Dict[str, float] = {}  # session_id -> last_active_ts

        # Measurements
        self._collapse_stats = {
            "governance_fires": 0,
            "governance_skips": 0,
            "telemetry_flushes": 0,
            "telemetry_suppressed": 0,
            "orchestration_coalesced": 0,
            "sync_coalesced": 0,
            "wakeup_reduction_pct": 0.0,
        }
        self._session_start = time.time()

        # Trace
        self._trace_dir = trace_dir
        self._trace_path = Path(trace_dir) / "orchestration_collapse_trace.jsonl" if trace_dir else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info("RuntimeCollapseCoordinator initialized | "
                          "gov_window=%d tokens | telem_flush=%.0fms",
                          self.GOVERNANCE_WINDOW_TOKENS,
                          self.TELEMETRY_FLUSH_INTERVAL_S * 1000)

    # -----------------------------------------------------------------------
    # Governance collapse
    # -----------------------------------------------------------------------

    def register_governance_callback(self, cb: Callable):
        """Register a governance function that should be called in batched windows."""
        self._governance_callbacks.append(cb)

    def token_generated(self, session_id: str, token_count: int = 1) -> bool:
        """
        Called once per token (or small batch). Returns True if governance
        should fire this window, False if it is suppressed (collapsed).
        Governance firing is batched to every GOVERNANCE_WINDOW_TOKENS tokens.
        """
        with self._lock:
            self._window_token_counter += token_count
            self._active_sessions[session_id] = time.perf_counter()

            if self._window_token_counter >= self.GOVERNANCE_WINDOW_TOKENS:
                # Fire governance window
                self._window_token_counter = 0
                self._governance_fires += 1
                self._collapse_stats["governance_fires"] += 1
                window_sessions = list(self._active_sessions.keys())

                # Async-friendly: just return True; caller invokes governance
                should_fire = True
            else:
                self._governance_skips += 1
                self._collapse_stats["governance_skips"] += 1
                should_fire = False

        if should_fire:
            self._trace_collapse_event("governance_window", {
                "tokens_since_last": self.GOVERNANCE_WINDOW_TOKENS,
                "active_sessions": len(self._active_sessions),
            })

        return should_fire

    def fire_governance(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronously invoke all registered governance callbacks in one pass."""
        results = {}
        ctx = context or {}
        t0 = time.perf_counter()
        for cb in self._governance_callbacks:
            try:
                result = cb(ctx)
                if result:
                    results.update(result if isinstance(result, dict) else {})
            except Exception as e:
                self._logger.debug("Governance callback error: %s", e)
        elapsed = time.perf_counter() - t0
        results["_governance_duration_ms"] = round(elapsed * 1000, 3)
        return results

    # -----------------------------------------------------------------------
    # Telemetry collapse
    # -----------------------------------------------------------------------

    def register_telemetry_flush_callback(self, cb: Callable):
        """Register a function that receives batched telemetry on flush."""
        self._telemetry_flush_callbacks.append(cb)

    def record_metric(self, key: str, value: Any):
        """
        Buffer a telemetry metric. Actual flush is deferred to flush interval.
        This replaces per-token metric emission with a batched buffer.
        """
        with self._lock:
            self._telemetry_buffer[key].append(value)
            self._telemetry_suppressed_count += 1

            now = time.perf_counter()
            if now - self._last_telemetry_flush >= self.TELEMETRY_FLUSH_INTERVAL_S:
                self._do_telemetry_flush_locked()

    def flush_telemetry(self):
        """Force a telemetry flush (call at end of batch or on demand)."""
        with self._lock:
            self._do_telemetry_flush_locked()

    def _do_telemetry_flush_locked(self):
        """Flush telemetry buffer — must be called with lock held."""
        if not self._telemetry_buffer:
            return
        # Aggregate: compute mean for numeric lists
        aggregated = {}
        for key, values in self._telemetry_buffer.items():
            if values and isinstance(values[0], (int, float)):
                aggregated[key] = round(sum(values) / len(values), 4)
                aggregated[key + "_count"] = len(values)
            else:
                aggregated[key] = values[-1]  # Last-value semantics for non-numeric
        self._telemetry_buffer.clear()

        suppressed = self._telemetry_suppressed_count
        self._telemetry_suppressed_count = 0
        self._last_telemetry_flush = time.perf_counter()
        self._collapse_stats["telemetry_flushes"] += 1
        self._collapse_stats["telemetry_suppressed"] += suppressed

        # Invoke flush callbacks
        for cb in self._telemetry_flush_callbacks:
            try:
                cb(aggregated)
            except Exception as e:
                self._logger.debug("Telemetry flush callback error: %s", e)

    # -----------------------------------------------------------------------
    # Orchestration coalescing
    # -----------------------------------------------------------------------

    def register_orchestration_callback(self, cb: Callable):
        """Register an orchestration task that benefits from coalescing."""
        self._orchestration_callbacks.append(cb)

    def request_orchestration(self, trigger: str, payload: Dict[str, Any] = None):
        """
        Submit an orchestration request. If a similar request was submitted
        within ORCHESTRATION_COALESCE_MS, it will be merged.
        """
        with self._lock:
            now = time.perf_counter()
            # Check if there's a recent pending request for same trigger
            for existing in self._pending_orchestration:
                if (existing["trigger"] == trigger and
                        (now - existing["ts"]) * 1000 < self.ORCHESTRATION_COALESCE_MS):
                    # Coalesce: merge payload, don't add new wakeup
                    if payload:
                        existing["payload"].update(payload)
                    self._orchestration_coalesced += 1
                    self._collapse_stats["orchestration_coalesced"] += 1
                    return

            # Add new pending request
            self._pending_orchestration.append({
                "trigger": trigger,
                "ts": now,
                "payload": payload or {},
            })

    def drain_orchestration(self) -> int:
        """
        Execute all pending orchestration work in one pass.
        Returns number of unique orchestration tasks executed.
        """
        with self._lock:
            tasks = list(self._pending_orchestration)
            self._pending_orchestration.clear()

        executed = 0
        for task in tasks:
            for cb in self._orchestration_callbacks:
                try:
                    cb(task["trigger"], task["payload"])
                except Exception as e:
                    self._logger.debug("Orchestration callback error: %s", e)
            executed += 1

        if executed:
            self._trace_collapse_event("orchestration_drain", {
                "tasks_executed": executed,
                "coalesced_total": self._orchestration_coalesced,
            })
        return executed

    # -----------------------------------------------------------------------
    # Synchronization cycle coalescing
    # -----------------------------------------------------------------------

    def add_sync_item(self, item: Any):
        """Add an item to the sync coalesce buffer."""
        with self._lock:
            self._sync_coalesce_buffer.append(item)
            if len(self._sync_coalesce_buffer) >= self.MAX_SYNC_COALESCE:
                self._sync_coalesced += len(self._sync_coalesce_buffer) - 1
                self._collapse_stats["sync_coalesced"] += len(self._sync_coalesce_buffer) - 1
                items = self._sync_coalesce_buffer.copy()
                self._sync_coalesce_buffer.clear()
                return items
        return None

    def flush_sync(self) -> List[Any]:
        """Flush all pending sync items at once."""
        with self._lock:
            items = self._sync_coalesce_buffer.copy()
            if items:
                self._sync_coalesced += max(0, len(items) - 1)
                self._collapse_stats["sync_coalesced"] += max(0, len(items) - 1)
            self._sync_coalesce_buffer.clear()
        return items

    # -----------------------------------------------------------------------
    # Collapse efficiency reporting
    # -----------------------------------------------------------------------

    def get_collapse_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._collapse_stats)
            total_gov_calls = stats["governance_fires"] + stats["governance_skips"]
            if total_gov_calls > 0:
                stats["wakeup_reduction_pct"] = round(
                    stats["governance_skips"] / total_gov_calls * 100, 1
                )
            stats["active_sessions"] = len(self._active_sessions)
            return stats

    def format_live_line(self) -> str:
        s = self.get_collapse_stats()
        return (
            f"[COLLAPSE] gov_fires={s['governance_fires']} "
            f"gov_skipped={s['governance_skips']} "
            f"wakeup_reduction={s['wakeup_reduction_pct']:.1f}% "
            f"telem_suppressed={s['telemetry_suppressed']} "
            f"orch_coalesced={s['orchestration_coalesced']} "
            f"sync_coalesced={s['sync_coalesced']}"
        )

    # -----------------------------------------------------------------------
    # Trace persistence
    # -----------------------------------------------------------------------

    def _trace_collapse_event(self, event_type: str, data: Dict[str, Any]):
        if not self._trace_path:
            return
        record = {"timestamp": time.time(), "event_type": event_type, **data}
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def emit_snapshot(self):
        """Emit a collapse statistics snapshot to trace."""
        stats = self.get_collapse_stats()
        self._trace_collapse_event("snapshot", stats)
