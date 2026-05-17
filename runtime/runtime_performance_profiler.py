"""
PRD Phase 41.0: Runtime Performance Profiler.
Measures TRUE end-to-end runtime cost using real timing instrumentation.
NO estimates. NO synthetic benchmarks.

Profiles:
- prefill latency
- decode latency
- token latency
- scheduler latency
- queue wait time
- batch formation latency
- stream delivery latency
- governance execution time
- telemetry overhead
- Python orchestration overhead
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
from collections import deque


class RuntimePerformanceProfiler:
    """
    PRD Phase 41.0: Real-time end-to-end runtime performance profiler.
    All measurements are wall-clock real timings captured at precise boundaries.
    """

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("PRD_RuntimeProfiler")

        # Live rolling windows (last 60 samples)
        self._prefill_latencies: deque = deque(maxlen=60)
        self._decode_latencies: deque = deque(maxlen=60)
        self._token_latencies: deque = deque(maxlen=60)
        self._scheduler_latencies: deque = deque(maxlen=60)
        self._queue_wait_times: deque = deque(maxlen=60)
        self._batch_formation_latencies: deque = deque(maxlen=60)
        self._stream_delivery_latencies: deque = deque(maxlen=60)
        self._governance_times: deque = deque(maxlen=60)
        self._telemetry_overheads: deque = deque(maxlen=60)
        self._orchestration_overheads: deque = deque(maxlen=60)

        # Per-request accumulator
        self._active_requests: Dict[str, Dict[str, float]] = {}

        self._trace_path = self.trace_dir / "runtime_timing_trace.jsonl"
        self._logger.info(f"RuntimePerformanceProfiler initialized → {self.trace_dir}")

    # -----------------------------------------------------------------------
    # Context-manager based timers (zero-overhead path)
    # -----------------------------------------------------------------------

    @contextmanager
    def time_prefill(self, request_id: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record("prefill", request_id, elapsed)

    @contextmanager
    def time_decode(self, request_id: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record("decode", request_id, elapsed)

    @contextmanager
    def time_scheduler(self, request_id: str = "global"):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record("scheduler", request_id, elapsed)

    @contextmanager
    def time_governance(self, request_id: str = "global"):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record("governance", request_id, elapsed)

    @contextmanager
    def time_telemetry(self, request_id: str = "global"):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record("telemetry", request_id, elapsed)

    @contextmanager
    def time_orchestration(self, request_id: str = "global"):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record("orchestration", request_id, elapsed)

    # -----------------------------------------------------------------------
    # Direct measurement APIs
    # -----------------------------------------------------------------------

    def record_token_latency(self, request_id: str, latency_sec: float):
        self._record("token", request_id, latency_sec)

    def record_queue_wait(self, request_id: str, wait_sec: float):
        self._record("queue_wait", request_id, wait_sec)

    def record_batch_formation(self, latency_sec: float):
        self._record("batch_formation", "global", latency_sec)

    def record_stream_delivery(self, request_id: str, latency_sec: float):
        self._record("stream_delivery", request_id, latency_sec)

    # -----------------------------------------------------------------------
    # Request lifecycle tracking
    # -----------------------------------------------------------------------

    def request_arrived(self, request_id: str):
        with self._lock:
            self._active_requests[request_id] = {
                "arrival_ts": time.perf_counter(),
                "prefill_sec": 0.0,
                "decode_sec": 0.0,
                "governance_sec": 0.0,
                "telemetry_sec": 0.0,
                "orchestration_sec": 0.0,
                "queue_wait_sec": 0.0,
                "tokens_generated": 0,
            }

    def request_token_generated(self, request_id: str):
        with self._lock:
            if request_id in self._active_requests:
                self._active_requests[request_id]["tokens_generated"] += 1

    def request_completed(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if request_id not in self._active_requests:
                return None
            r = self._active_requests.pop(request_id)
            total_sec = time.perf_counter() - r["arrival_ts"]
            tokens = max(r["tokens_generated"], 1)
            record = {
                "timestamp": time.time(),
                "request_id": request_id,
                "total_sec": round(total_sec, 4),
                "prefill_sec": round(r["prefill_sec"], 4),
                "decode_sec": round(r["decode_sec"], 4),
                "governance_sec": round(r["governance_sec"], 4),
                "telemetry_sec": round(r["telemetry_sec"], 4),
                "orchestration_sec": round(r["orchestration_sec"], 4),
                "queue_wait_sec": round(r["queue_wait_sec"], 4),
                "tokens_generated": tokens,
                "tokens_per_sec": round(tokens / total_sec, 2) if total_sec > 0 else 0,
                "governance_overhead_pct": round(r["governance_sec"] / total_sec * 100, 1) if total_sec > 0 else 0,
                "telemetry_overhead_pct": round(r["telemetry_sec"] / total_sec * 100, 1) if total_sec > 0 else 0,
                "orchestration_overhead_pct": round(r["orchestration_sec"] / total_sec * 100, 1) if total_sec > 0 else 0,
            }
        self._persist(record)
        return record

    # -----------------------------------------------------------------------
    # Live summary
    # -----------------------------------------------------------------------

    def get_live_summary(self) -> Dict[str, Any]:
        def safe_avg(dq: deque) -> float:
            return round(sum(dq) / len(dq), 4) if dq else 0.0

        def safe_p99(dq: deque) -> float:
            if not dq:
                return 0.0
            srt = sorted(dq)
            idx = max(0, int(len(srt) * 0.99) - 1)
            return round(srt[idx], 4)

        return {
            "avg_prefill_sec": safe_avg(self._prefill_latencies),
            "avg_decode_sec": safe_avg(self._decode_latencies),
            "avg_token_latency_ms": round(safe_avg(self._token_latencies) * 1000, 2),
            "p99_token_latency_ms": round(safe_p99(self._token_latencies) * 1000, 2),
            "avg_scheduler_sec": safe_avg(self._scheduler_latencies),
            "avg_queue_wait_sec": safe_avg(self._queue_wait_times),
            "avg_batch_formation_sec": safe_avg(self._batch_formation_latencies),
            "avg_stream_delivery_sec": safe_avg(self._stream_delivery_latencies),
            "avg_governance_sec": safe_avg(self._governance_times),
            "avg_telemetry_sec": safe_avg(self._telemetry_overheads),
            "avg_orchestration_sec": safe_avg(self._orchestration_overheads),
            "active_requests": len(self._active_requests),
        }

    def format_live_line(self) -> str:
        s = self.get_live_summary()
        return (
            f"[PROFILER] "
            f"prefill={s['avg_prefill_sec']*1000:.1f}ms "
            f"decode={s['avg_decode_sec']*1000:.1f}ms "
            f"tok={s['avg_token_latency_ms']:.1f}ms "
            f"sched={s['avg_scheduler_sec']*1000:.1f}ms "
            f"q_wait={s['avg_queue_wait_sec']*1000:.1f}ms "
            f"gov={s['avg_governance_sec']*1000:.1f}ms "
            f"telem={s['avg_telemetry_sec']*1000:.1f}ms "
            f"orch={s['avg_orchestration_sec']*1000:.1f}ms"
        )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _record(self, category: str, request_id: str, elapsed: float):
        with self._lock:
            # Update rolling window
            bucket_map = {
                "prefill": self._prefill_latencies,
                "decode": self._decode_latencies,
                "token": self._token_latencies,
                "scheduler": self._scheduler_latencies,
                "queue_wait": self._queue_wait_times,
                "batch_formation": self._batch_formation_latencies,
                "stream_delivery": self._stream_delivery_latencies,
                "governance": self._governance_times,
                "telemetry": self._telemetry_overheads,
                "orchestration": self._orchestration_overheads,
            }
            if category in bucket_map:
                bucket_map[category].append(elapsed)

            # Accumulate into active request if present
            accu_map = {
                "prefill": "prefill_sec",
                "decode": "decode_sec",
                "governance": "governance_sec",
                "telemetry": "telemetry_sec",
                "orchestration": "orchestration_sec",
                "queue_wait": "queue_wait_sec",
            }
            if request_id in self._active_requests and category in accu_map:
                self._active_requests[request_id][accu_map[category]] += elapsed

    def _persist(self, record: Dict[str, Any]):
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Trace persistence error: {e}")
