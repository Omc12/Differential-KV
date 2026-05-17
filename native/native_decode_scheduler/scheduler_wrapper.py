"""
native_decode_scheduler Python wrapper.
RCO-N Phase 41.1

Provides a Python-compatible interface to the native C++ NativeDecodeScheduler.
Falls back to a pure-Python implementation if the native extension is not compiled.

Usage:
    from native.native_decode_scheduler.scheduler_wrapper import DecodeScheduler
    sched = DecodeScheduler(max_batch_size=32)
    sched.admit("session_1", "req_1", max_tokens=128)
    batch = sched.prepare_batch()
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
import heapq

log = logging.getLogger("NativeDecodeScheduler")

# -----------------------------------------------------------------------
# Attempt native import
# -----------------------------------------------------------------------
_NATIVE_AVAILABLE = False
_native_mod = None

try:
    import importlib.util
    import sys
    # Look for compiled extension in current directory or build/
    _search_paths = [
        Path(__file__).parent,
        Path(__file__).parent / "build",
        Path(__file__).parent / "Release",
        Path(__file__).parent / "Debug",
    ]
    for _p in _search_paths:
        for _ext in [".pyd", ".so", ".dylib"]:
            _candidates = list(_p.glob(f"native_decode_scheduler*{_ext}"))
            if _candidates:
                _spec = importlib.util.spec_from_file_location(
                    "native_decode_scheduler", _candidates[0]
                )
                _native_mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_native_mod)
                _NATIVE_AVAILABLE = True
                log.info("Native decode scheduler loaded from: %s", _candidates[0])
                break
        if _NATIVE_AVAILABLE:
            break
except Exception as _e:
    log.debug("Native decode scheduler not available: %s — using Python fallback", _e)


# -----------------------------------------------------------------------
# Pure Python fallback (mirrors C++ interface)
# -----------------------------------------------------------------------

class _PySlot:
    __slots__ = ["session_id", "request_id", "tokens_generated", "max_tokens",
                 "priority", "finished", "admitted_ts", "last_token_ts"]

    def __init__(self, session_id, request_id, max_tokens, priority):
        self.session_id = session_id
        self.request_id = request_id
        self.tokens_generated = 0
        self.max_tokens = max_tokens
        self.priority = priority
        self.finished = False
        self.admitted_ts = time.perf_counter()
        self.last_token_ts = time.perf_counter()


class _PythonFallbackScheduler:
    """Pure-Python decode scheduler — mirrors NativeDecodeScheduler interface."""

    def __init__(self, max_batch_size: int = 32, starvation_threshold_ms: float = 1.5):
        self._lock = threading.Lock()
        self._max_batch_size = max_batch_size
        self._starvation_threshold_ms = starvation_threshold_ms
        self._slots: Dict[str, _PySlot] = {}
        self._admission_heap: List = []  # (-priority, counter, slot)
        self._counter = 0

        # Stats
        self._total_batch_steps = 0
        self._total_tokens = 0
        self._slot_fills = 0
        self._slot_evictions = 0
        self._starvation_events = 0
        self._starvation_gap_sum = 0.0
        self._max_starvation_gap = 0.0
        self._batch_size_sum = 0.0
        self._overhead_sum_us = 0.0
        self._last_step_end_ts = time.perf_counter()
        self._first_step = True

    def admit(self, session_id: str, request_id: str, max_tokens: int = 128, priority: int = 0):
        slot = _PySlot(session_id, request_id, max_tokens, priority)
        with self._lock:
            self._counter += 1
            heapq.heappush(self._admission_heap, (-priority, self._counter, slot))

    def complete(self, session_id: str):
        with self._lock:
            if session_id in self._slots:
                self._slots[session_id].finished = True

    def cancel(self, session_id: str):
        self.complete(session_id)

    def prepare_batch(self) -> List[str]:
        t0 = time.perf_counter()
        with self._lock:
            # Evict finished
            finished = [sid for sid, s in self._slots.items() if s.finished]
            for sid in finished:
                del self._slots[sid]
                self._slot_evictions += 1

            # Fill from admission queue
            while self._admission_heap and len(self._slots) < self._max_batch_size:
                _, _, slot = heapq.heappop(self._admission_heap)
                self._slots[slot.session_id] = slot
                self._slot_fills += 1

            batch = [sid for sid, s in self._slots.items() if not s.finished]
            self._total_batch_steps += 1
            self._batch_size_sum += len(batch)
            self._overhead_sum_us += (time.perf_counter() - t0) * 1e6

        return batch

    def record_token(self, session_id: str, count: int = 1):
        with self._lock:
            slot = self._slots.get(session_id)
            if slot:
                slot.tokens_generated += count
                slot.last_token_ts = time.perf_counter()
                self._total_tokens += count
                if slot.tokens_generated >= slot.max_tokens:
                    slot.finished = True

    def step_begin(self):
        if not self._first_step:
            now = time.perf_counter()
            gap_ms = (now - self._last_step_end_ts) * 1000
            if gap_ms > self._starvation_threshold_ms:
                self._starvation_events += 1
                self._starvation_gap_sum += gap_ms
                if gap_ms > self._max_starvation_gap:
                    self._max_starvation_gap = gap_ms
        self._first_step = False

    def step_end(self, batch: List[str]):
        self._last_step_end_ts = time.perf_counter()

    def get_stats_json(self) -> str:
        with self._lock:
            avg_batch = self._batch_size_sum / max(self._total_batch_steps, 1)
            avg_gap = self._starvation_gap_sum / max(self._starvation_events, 1)
            avg_overhead = self._overhead_sum_us / max(self._total_batch_steps, 1)
            return json.dumps({
                "total_batch_steps": self._total_batch_steps,
                "total_tokens_scheduled": self._total_tokens,
                "slot_fills": self._slot_fills,
                "slot_evictions": self._slot_evictions,
                "starvation_events": self._starvation_events,
                "active_slots": len(self._slots),
                "admission_queue_depth": len(self._admission_heap),
                "avg_batch_size": round(avg_batch, 2),
                "avg_starvation_gap_ms": round(avg_gap, 3),
                "max_starvation_gap_ms": round(self._max_starvation_gap, 3),
                "scheduler_overhead_us": round(avg_overhead, 3),
                "backend": "python_fallback",
            })


# -----------------------------------------------------------------------
# Public interface — uses native if available, else Python fallback
# -----------------------------------------------------------------------

class DecodeScheduler:
    """
    Public interface to the native (or fallback) decode scheduler.
    Adds trace persistence and live monitoring on top.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        starvation_threshold_ms: float = 1.5,
        trace_dir: Optional[Path] = None,
    ):
        if _NATIVE_AVAILABLE:
            self._sched = _native_mod.NativeDecodeScheduler(
                max_batch_size, starvation_threshold_ms
            )
            self._backend = "native_cpp"
        else:
            self._sched = _PythonFallbackScheduler(max_batch_size, starvation_threshold_ms)
            self._backend = "python_fallback"

        self._trace_path = Path(trace_dir) / "native_scheduler_trace.jsonl" if trace_dir else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._snapshot_interval = 5.0
        self._last_snapshot_ts = time.time()

        log.info("DecodeScheduler initialized | backend=%s | max_batch=%d",
                 self._backend, max_batch_size)

    # -----------------------------------------------------------------------
    # Forwarded interface
    # -----------------------------------------------------------------------
    def admit(self, session_id: str, request_id: str, max_tokens: int = 128, priority: int = 0):
        self._sched.admit(session_id, request_id, max_tokens, priority)

    def complete(self, session_id: str):
        self._sched.complete(session_id)

    def cancel(self, session_id: str):
        self._sched.cancel(session_id)

    def prepare_batch(self) -> List[str]:
        return self._sched.prepare_batch()

    def record_token(self, session_id: str, count: int = 1):
        self._sched.record_token(session_id, count)

    def step_begin(self):
        self._sched.step_begin()

    def step_end(self, batch: List[str]):
        self._sched.step_end(batch)

    # -----------------------------------------------------------------------
    # Stats and trace
    # -----------------------------------------------------------------------
    def get_stats_json(self) -> str:
        return self._sched.get_stats_json()

    def get_stats(self) -> Dict[str, Any]:
        return json.loads(self.get_stats_json())

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def native_available(self) -> bool:
        return _NATIVE_AVAILABLE

    def maybe_emit_trace(self):
        now = time.time()
        if now - self._last_snapshot_ts < self._snapshot_interval:
            return
        self._last_snapshot_ts = now
        if self._trace_path:
            try:
                stats = self.get_stats()
                stats["timestamp"] = now
                stats["backend"] = self._backend
                with open(self._trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(stats) + "\n")
            except Exception:
                pass

    def format_live_line(self) -> str:
        s = self.get_stats()
        return (
            f"[NATIVE_SCHED/{self._backend}] "
            f"active={s.get('active_slots', 0)} "
            f"steps={s.get('total_batch_steps', 0)} "
            f"starvation={s.get('starvation_events', 0)} "
            f"overhead={s.get('scheduler_overhead_us', 0):.2f}us "
            f"avg_batch={s.get('avg_batch_size', 0):.1f}"
        )
