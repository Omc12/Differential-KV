"""
PRD Phase 41.0: Scheduler & Queue Turbulence Profiler.
Measures queue inefficiency, scheduler contention, batch fragmentation,
cancellation overhead, reconnect overhead, and stream synchronization delays.

Operational orchestration overhead may dominate runtime cost.
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
from contextlib import contextmanager


class SchedulerQueueTurbulenceProfiler:
    """
    PRD Phase 41.0: Profiles queue and scheduler turbulence in real time.
    Measures the operational overhead of request lifecycle management.
    """

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("PRD_QueueProfiler")

        # Queue depth tracking
        self._queue_depth_history: deque = deque(maxlen=300)
        self._queue_depth_timestamps: deque = deque(maxlen=300)
        self._current_queue_depth: int = 0

        # Batch fragmentation tracking
        self._batch_sizes: deque = deque(maxlen=200)
        self._ideal_batch_size: int = 8  # expected optimal
        self._fragmented_batches: int = 0  # batches with size < 50% of ideal

        # Latency tracking
        self._enqueue_latencies: deque = deque(maxlen=200)  # time to add to queue (ms)
        self._dequeue_latencies: deque = deque(maxlen=200)  # time to pull from queue (ms)
        self._scheduler_decision_latencies: deque = deque(maxlen=200)
        self._cancellation_latencies: deque = deque(maxlen=100)
        self._reconnect_latencies: deque = deque(maxlen=100)
        self._stream_sync_latencies: deque = deque(maxlen=200)

        # Event counters
        self._cancellation_count: int = 0
        self._reconnect_count: int = 0
        self._preemption_count: int = 0
        self._scheduler_contention_events: int = 0

        # In-flight request tracking
        self._enqueue_timestamps: Dict[str, float] = {}

        self._trace_path = self.trace_dir / "queue_turbulence_trace.jsonl"
        self._logger.info(f"SchedulerQueueTurbulenceProfiler initialized → {self.trace_dir}")

    # -----------------------------------------------------------------------
    # Queue operations
    # -----------------------------------------------------------------------

    def request_enqueued(self, request_id: str, queue_depth: int):
        t_enqueue = time.perf_counter()
        with self._lock:
            self._enqueue_timestamps[request_id] = t_enqueue
            self._current_queue_depth = queue_depth
            self._queue_depth_history.append(queue_depth)
            self._queue_depth_timestamps.append(time.time())
        self._persist_event("enqueue", {
            "request_id": request_id,
            "queue_depth": queue_depth,
        })

    def request_dequeued(self, request_id: str, queue_depth: int):
        t_dequeue = time.perf_counter()
        with self._lock:
            enq_ts = self._enqueue_timestamps.pop(request_id, None)
            self._current_queue_depth = queue_depth
            self._queue_depth_history.append(queue_depth)
            self._queue_depth_timestamps.append(time.time())

            if enq_ts is not None:
                wait_sec = t_dequeue - enq_ts
                self._dequeue_latencies.append(wait_sec)
        self._persist_event("dequeue", {
            "request_id": request_id,
            "queue_depth": queue_depth,
        })

    # -----------------------------------------------------------------------
    # Batch events
    # -----------------------------------------------------------------------

    def batch_formed(self, batch_size: int, intended_size: int):
        """Record a batch formation event."""
        with self._lock:
            self._batch_sizes.append(batch_size)
            if batch_size < intended_size * 0.5:
                self._fragmented_batches += 1
        self._persist_event("batch_formed", {
            "batch_size": batch_size,
            "intended_size": intended_size,
            "fragmented": batch_size < intended_size * 0.5,
        })

    # -----------------------------------------------------------------------
    # Context-manager timers
    # -----------------------------------------------------------------------

    @contextmanager
    def time_scheduler_decision(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._scheduler_decision_latencies.append(elapsed)

    @contextmanager
    def time_cancellation(self, request_id: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._cancellation_latencies.append(elapsed)
                self._cancellation_count += 1
            self._persist_event("cancellation", {
                "request_id": request_id,
                "duration_ms": round(elapsed * 1000, 3),
            })

    @contextmanager
    def time_reconnect(self, session_id: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._reconnect_latencies.append(elapsed)
                self._reconnect_count += 1
            self._persist_event("reconnect", {
                "session_id": session_id,
                "duration_ms": round(elapsed * 1000, 3),
            })

    @contextmanager
    def time_stream_sync(self, request_id: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                self._stream_sync_latencies.append(elapsed)
            self._persist_event("stream_sync", {
                "request_id": request_id,
                "duration_ms": round(elapsed * 1000, 3),
            })

    # -----------------------------------------------------------------------
    # Contention / preemption recording
    # -----------------------------------------------------------------------

    def record_scheduler_contention(self, reason: str = "unknown"):
        with self._lock:
            self._scheduler_contention_events += 1
        self._persist_event("contention", {"reason": reason})

    def record_preemption(self, request_id: str, reason: str = "unknown"):
        with self._lock:
            self._preemption_count += 1
        self._persist_event("preemption", {"request_id": request_id, "reason": reason})

    # -----------------------------------------------------------------------
    # Live reporting
    # -----------------------------------------------------------------------

    def get_live_summary(self) -> Dict[str, Any]:
        with self._lock:
            def avg_ms(dq: deque) -> float:
                return round(sum(dq) / len(dq) * 1000, 2) if dq else 0.0
            def p95_ms(dq: deque) -> float:
                if not dq:
                    return 0.0
                srt = sorted(dq)
                return round(srt[int(len(srt) * 0.95)] * 1000, 2)

            avg_q = round(sum(self._queue_depth_history) / len(self._queue_depth_history), 1) if self._queue_depth_history else 0.0
            max_q = max(self._queue_depth_history) if self._queue_depth_history else 0
            total_batches = len(self._batch_sizes)
            frag_rate = round(self._fragmented_batches / total_batches, 3) if total_batches > 0 else 0.0
            avg_batch = round(sum(self._batch_sizes) / len(self._batch_sizes), 1) if self._batch_sizes else 0.0

        return {
            "current_queue_depth": self._current_queue_depth,
            "avg_queue_depth": avg_q,
            "max_queue_depth": max_q,
            "avg_queue_wait_ms": avg_ms(self._dequeue_latencies),
            "p95_queue_wait_ms": p95_ms(self._dequeue_latencies),
            "avg_scheduler_decision_ms": avg_ms(self._scheduler_decision_latencies),
            "avg_batch_size": avg_batch,
            "batch_fragmentation_rate": frag_rate,
            "fragmented_batches": self._fragmented_batches,
            "cancellation_count": self._cancellation_count,
            "avg_cancellation_ms": avg_ms(self._cancellation_latencies),
            "reconnect_count": self._reconnect_count,
            "avg_reconnect_ms": avg_ms(self._reconnect_latencies),
            "avg_stream_sync_ms": avg_ms(self._stream_sync_latencies),
            "scheduler_contention_events": self._scheduler_contention_events,
            "preemption_count": self._preemption_count,
        }

    def format_live_line(self) -> str:
        s = self.get_live_summary()
        return (
            f"[QUEUE] depth={s['current_queue_depth']} "
            f"wait={s['avg_queue_wait_ms']:.1f}ms "
            f"sched={s['avg_scheduler_decision_ms']:.1f}ms "
            f"frag={s['batch_fragmentation_rate']:.1%} "
            f"cancel={s['cancellation_count']} "
            f"reconnect={s['reconnect_count']} "
            f"contention={s['scheduler_contention_events']}"
        )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _persist_event(self, event_type: str, data: Dict[str, Any]):
        record = {"timestamp": time.time(), "event_type": event_type, **data}
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Queue turbulence trace error: {e}")
