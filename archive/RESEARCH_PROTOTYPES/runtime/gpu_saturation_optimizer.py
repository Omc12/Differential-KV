"""
RCO-N Phase 41.1: GPU Saturation Optimizer.

Maximizes continuous GPU utilization by tracking idle gaps,
kernel starvation intervals, and synchronization stalls.
Coordinates scheduler pacing to maintain sustained occupancy.

TARGET: Material occupancy increase by eliminating idle gaps.
"""

import time
import threading
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from collections import deque


class SaturationWindow:
    """A time-bounded measurement window for GPU saturation tracking."""
    def __init__(self, window_sec: float = 10.0):
        self._window_sec = window_sec
        self._events: deque = deque()   # (ts, is_active)

    def record(self, is_active: bool):
        now = time.perf_counter()
        self._events.append((now, is_active))
        self._purge(now)

    def _purge(self, now: float):
        cutoff = now - self._window_sec
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def saturation_ratio(self) -> float:
        if not self._events:
            return 0.0
        active = sum(1 for _, is_active in self._events if is_active)
        return active / len(self._events)


class GPUSaturationOptimizer:
    """
    RCO-N Phase 41.1: GPU saturation optimizer.

    Tracks every GPU active/idle transition and computes:
    - saturation ratio (% of time GPU is doing work)
    - starvation gap distribution
    - kernel dispatch frequency
    - synchronization stall contribution

    Feeds pacing signals back to the scheduler to eliminate idle cycles.
    """

    STARVATION_GAP_MS           = 1.5   # Gaps > 1.5ms are starvation events
    TARGET_SATURATION_RATIO     = 0.80  # 80% sustained occupancy target
    PACING_SIGNAL_INTERVAL_S    = 0.25  # Emit pacing signals every 250ms
    KERNEL_TRACE_WINDOW_S       = 30.0  # Rolling kernel window

    def __init__(self, trace_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._logger = logging.getLogger("RCO_GPUSaturationOptimizer")

        # Saturation tracking
        self._saturation_window = SaturationWindow(window_sec=10.0)
        self._last_kernel_ts = time.perf_counter()
        self._gpu_active = False

        # Gap tracking
        self._idle_gap_start: Optional[float] = None
        self._starvation_gaps: deque = deque(maxlen=500)   # gap durations in ms
        self._starvation_event_count = 0

        # Kernel dispatch tracking
        self._kernel_count = 0
        self._kernel_timestamps: deque = deque(maxlen=1000)

        # Sync stall tracking
        self._sync_stall_durations: deque = deque(maxlen=200)
        self._total_sync_stall_ms = 0.0

        # Pacing callbacks (scheduler calls these to pace work issuance)
        self._pacing_callbacks: List[Callable] = []
        self._last_pacing_signal_ts = time.perf_counter()
        self._pacing_signals_emitted = 0

        # Session and telemetry
        self._session_start = time.time()
        self._occupancy_history: deque = deque(maxlen=120)  # 2 min at 1Hz
        self._last_occupancy_sample_ts = time.perf_counter()

        self._trace_path = Path(trace_dir) / "gpu_saturation_trace.jsonl" if trace_dir else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "GPUSaturationOptimizer initialized | "
            "starvation_threshold=%.1fms | target_saturation=%.0f%%",
            self.STARVATION_GAP_MS, self.TARGET_SATURATION_RATIO * 100
        )

    # -----------------------------------------------------------------------
    # GPU activity signaling (call from decode hot-path)
    # -----------------------------------------------------------------------

    def kernel_dispatched(self, kernel_name: str = "decode", batch_size: int = 1):
        """Signal that a GPU kernel was just dispatched."""
        now = time.perf_counter()
        with self._lock:
            # If we were idle, record the gap
            if self._idle_gap_start is not None:
                gap_ms = (now - self._idle_gap_start) * 1000
                if gap_ms > self.STARVATION_GAP_MS:
                    self._starvation_gaps.append(gap_ms)
                    self._starvation_event_count += 1
                self._idle_gap_start = None

            self._gpu_active = True
            self._last_kernel_ts = now
            self._kernel_count += 1
            self._kernel_timestamps.append(now)

        self._saturation_window.record(True)
        self._maybe_emit_pacing_signal()
        self._maybe_sample_occupancy(now)

    def kernel_completed(self):
        """Signal that the last dispatched GPU kernel completed."""
        now = time.perf_counter()
        with self._lock:
            self._gpu_active = False
            self._idle_gap_start = now
        self._saturation_window.record(False)

    def record_sync_stall(self, stall_ms: float):
        """Record a CUDA synchronization stall."""
        with self._lock:
            self._sync_stall_durations.append(stall_ms)
            self._total_sync_stall_ms += stall_ms

    def mark_decode_step_start(self):
        """Call at the start of each decode step."""
        self.kernel_dispatched("decode_step")

    def mark_decode_step_end(self):
        """Call at the end of each decode step."""
        self.kernel_completed()

    # -----------------------------------------------------------------------
    # Pacing feedback
    # -----------------------------------------------------------------------

    def register_pacing_callback(self, cb: Callable):
        """
        Register a callback that receives pacing signals.
        The scheduler uses these to decide when to issue work.
        """
        self._pacing_callbacks.append(cb)

    def _maybe_emit_pacing_signal(self):
        now = time.perf_counter()
        if (now - self._last_pacing_signal_ts) < self.PACING_SIGNAL_INTERVAL_S:
            return

        self._last_pacing_signal_ts = now
        sat_ratio = self._saturation_window.saturation_ratio()
        signal = {
            "saturation_ratio": round(sat_ratio, 4),
            "below_target": sat_ratio < self.TARGET_SATURATION_RATIO,
            "starvation_events_recent": min(len(self._starvation_gaps), 20),
            "recommendation": "increase_batch" if sat_ratio < self.TARGET_SATURATION_RATIO else "stable",
        }
        self._pacing_signals_emitted += 1

        for cb in self._pacing_callbacks:
            try:
                cb(signal)
            except Exception:
                pass

    def _maybe_sample_occupancy(self, now: float):
        if (now - self._last_occupancy_sample_ts) < 1.0:  # 1Hz sample
            return
        self._last_occupancy_sample_ts = now
        sat = self._saturation_window.saturation_ratio()
        with self._lock:
            self._occupancy_history.append((time.time(), sat))

        if self._trace_path:
            self._persist_sample(sat)

    # -----------------------------------------------------------------------
    # Optimization analysis
    # -----------------------------------------------------------------------

    def get_saturation_stats(self) -> Dict[str, Any]:
        with self._lock:
            sat_ratio = self._saturation_window.saturation_ratio()
            recent_gaps = list(self._starvation_gaps)[-20:]
            avg_gap = round(sum(recent_gaps) / len(recent_gaps), 2) if recent_gaps else 0.0
            max_gap = round(max(recent_gaps), 2) if recent_gaps else 0.0

            # Kernel dispatch rate (last 10 seconds)
            now = time.perf_counter()
            recent_kernels = sum(1 for t in self._kernel_timestamps if t > now - 10.0)
            kernel_hz = round(recent_kernels / 10.0, 1)

            avg_sync_stall = (
                round(sum(self._sync_stall_durations) / len(self._sync_stall_durations), 2)
                if self._sync_stall_durations else 0.0
            )

        return {
            "saturation_ratio": round(sat_ratio, 4),
            "saturation_pct": round(sat_ratio * 100, 1),
            "target_saturation_pct": round(self.TARGET_SATURATION_RATIO * 100, 1),
            "at_target": sat_ratio >= self.TARGET_SATURATION_RATIO,
            "starvation_events": self._starvation_event_count,
            "avg_starvation_gap_ms": avg_gap,
            "max_starvation_gap_ms": max_gap,
            "kernel_dispatch_hz": kernel_hz,
            "total_kernels": self._kernel_count,
            "avg_sync_stall_ms": avg_sync_stall,
            "total_sync_stall_ms": round(self._total_sync_stall_ms, 1),
            "pacing_signals_emitted": self._pacing_signals_emitted,
        }

    def format_live_line(self) -> str:
        s = self.get_saturation_stats()
        at_target = "[OK]" if s["at_target"] else "[LOW]"
        return (
            f"[GPU_SAT] {at_target} "
            f"sat={s['saturation_pct']:.1f}% "
            f"target={s['target_saturation_pct']:.0f}% "
            f"starvation={s['starvation_events']} "
            f"avg_gap={s['avg_starvation_gap_ms']:.1f}ms "
            f"kernel_hz={s['kernel_dispatch_hz']:.1f} "
            f"sync_stall={s['avg_sync_stall_ms']:.1f}ms"
        )

    def _persist_sample(self, saturation: float):
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "saturation_ratio": round(saturation, 4),
                    "starvation_events": self._starvation_event_count,
                    "kernel_count": self._kernel_count,
                }) + "\n")
        except Exception:
            pass
