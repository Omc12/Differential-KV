"""
native_telemetry_counter_layer Python wrapper.
RCO-N Phase 41.1

Provides a Python interface to the native C++ NativeTelemetryCounterLayer.
Falls back to a pure-Python implementation if the native extension is not compiled.
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger("NativeTelemetryWrapper")

_NATIVE_AVAILABLE = False
_native_mod = None

try:
    import importlib.util
    _search_paths = [
        Path(__file__).parent,
        Path(__file__).parent / "build",
        Path(__file__).parent / "Release",
        Path(__file__).parent / "Debug",
    ]
    for _p in _search_paths:
        for _ext in [".pyd", ".so", ".dylib"]:
            _candidates = list(_p.glob(f"native_telemetry_counter_layer*{_ext}"))
            if _candidates:
                _spec = importlib.util.spec_from_file_location(
                    "native_telemetry_counter_layer", _candidates[0]
                )
                _native_mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_native_mod)
                _NATIVE_AVAILABLE = True
                log.info("Native telemetry counter layer loaded from: %s", _candidates[0])
                break
        if _NATIVE_AVAILABLE:
            break
except Exception as _e:
    log.debug("Native telemetry counter layer not available: %s — using Python fallback", _e)


class _PythonFallbackTelemetryCounters:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset_counters()

    def reset_counters(self):
        with self._lock:
            self._gpu_kernels_dispatched = 0
            self._gpu_starvation_events = 0
            self._gpu_sync_stalls = 0
            self._gpu_total_stall_ms = 0.0
            self._scheduler_steps = 0
            self._scheduler_admissions = 0
            self._scheduler_evictions = 0
            self._queue_enqueues = 0
            self._queue_dequeues = 0
            self._queue_cancellations = 0
            self._queue_reconnects = 0
            self._queue_reconnects_coalesced = 0
            self._tokens_generated = 0
            self._governance_fires = 0
            self._governance_skips = 0
            self._dense_fallbacks = 0
            self._partial_repairs = 0
            self._fusion_calls = 0
            self._telemetry_suppressed = 0
            self._telemetry_emitted = 0
            self._created_ts = time.perf_counter()

    # hot-path Increments
    def gpu_kernel_dispatched(self):
        with self._lock: self._gpu_kernels_dispatched += 1

    def gpu_starvation_event(self):
        with self._lock: self._gpu_starvation_events += 1

    def gpu_sync_stall(self, ms: float):
        with self._lock:
            self._gpu_sync_stalls += 1
            self._gpu_total_stall_ms += ms

    def scheduler_step(self):
        with self._lock: self._scheduler_steps += 1

    def scheduler_admission(self):
        with self._lock: self._scheduler_admissions += 1

    def scheduler_eviction(self):
        with self._lock: self._scheduler_evictions += 1

    def queue_enqueue(self):
        with self._lock: self._queue_enqueues += 1

    def queue_dequeue(self):
        with self._lock: self._queue_dequeues += 1

    def queue_cancel(self):
        with self._lock: self._queue_cancellations += 1

    def queue_reconnect(self, coalesced: bool):
        with self._lock:
            self._queue_reconnects += 1
            if coalesced:
                self._queue_reconnects_coalesced += 1

    def token_generated(self, n: int = 1):
        with self._lock: self._tokens_generated += n

    def governance_fired(self):
        with self._lock: self._governance_fires += 1

    def governance_skipped(self):
        with self._lock: self._governance_skips += 1

    def dense_fallback(self):
        with self._lock: self._dense_fallbacks += 1

    def partial_repair(self):
        with self._lock: self._partial_repairs += 1

    def fusion_call(self):
        with self._lock: self._fusion_calls += 1

    def telemetry_suppressed(self):
        with self._lock: self._telemetry_suppressed += 1

    def telemetry_emitted(self):
        with self._lock: self._telemetry_emitted += 1

    def governance_collapse_ratio(self) -> float:
        with self._lock:
            tot = self._governance_fires + self._governance_skips
            return self._governance_skips / tot if tot > 0 else 0.0

    def queue_reconnect_coalesce_ratio(self) -> float:
        with self._lock:
            tot = self._queue_reconnects
            return self._queue_reconnects_coalesced / tot if tot > 0 else 0.0

    def telemetry_suppression_ratio(self) -> float:
        with self._lock:
            tot = self._telemetry_suppressed + self._telemetry_emitted
            return self._telemetry_suppressed / tot if tot > 0 else 0.0

    def get_snapshot_json(self) -> str:
        with self._lock:
            elapsed_ms = (time.perf_counter() - self._created_ts) * 1000.0
            gov = self.governance_collapse_ratio()
            telem = self.telemetry_suppression_ratio()
            q_rec = self.queue_reconnect_coalesce_ratio()
            return json.dumps({
                "elapsed_ms": elapsed_ms,
                "gpu_kernels_dispatched": self._gpu_kernels_dispatched,
                "gpu_starvation_events": self._gpu_starvation_events,
                "gpu_sync_stalls": self._gpu_sync_stalls,
                "gpu_total_stall_ms": self._gpu_total_stall_ms,
                "scheduler_steps": self._scheduler_steps,
                "scheduler_admissions": self._scheduler_admissions,
                "scheduler_evictions": self._scheduler_evictions,
                "queue_enqueues": self._queue_enqueues,
                "queue_dequeues": self._queue_dequeues,
                "queue_cancellations": self._queue_cancellations,
                "queue_reconnects": self._queue_reconnects,
                "queue_reconnects_coalesced": self._queue_reconnects_coalesced,
                "tokens_generated": self._tokens_generated,
                "governance_fires": self._governance_fires,
                "governance_skips": self._governance_skips,
                "dense_fallbacks": self._dense_fallbacks,
                "partial_repairs": self._partial_repairs,
                "fusion_calls": self._fusion_calls,
                "telemetry_suppressed": self._telemetry_suppressed,
                "telemetry_emitted": self._telemetry_emitted,
                "governance_collapse_ratio": round(gov, 4),
                "telemetry_suppress_ratio": round(telem, 4),
                "reconnect_coalesce_ratio": round(q_rec, 4),
                "backend": "python_fallback",
            })


class TelemetryCounters:
    def __init__(self):
        if _NATIVE_AVAILABLE:
            self._layer = _native_mod.NativeTelemetryCounterLayer()
            self._backend = "native_cpp"
        else:
            self._layer = _PythonFallbackTelemetryCounters()
            self._backend = "python_fallback"

    def gpu_kernel_dispatched(self):
        self._layer.gpu_kernel_dispatched()

    def gpu_starvation_event(self):
        self._layer.gpu_starvation_event()

    def gpu_sync_stall(self, ms: float):
        self._layer.gpu_sync_stall(ms)

    def scheduler_step(self):
        self._layer.scheduler_step()

    def scheduler_admission(self):
        self._layer.scheduler_admission()

    def scheduler_eviction(self):
        self._layer.scheduler_eviction()

    def queue_enqueue(self):
        self._layer.queue_enqueue()

    def queue_dequeue(self):
        self._layer.queue_dequeue()

    def queue_cancel(self):
        self._layer.queue_cancel()

    def queue_reconnect(self, coalesced: bool):
        self._layer.queue_reconnect(coalesced)

    def token_generated(self, n: int = 1):
        self._layer.token_generated(n)

    def governance_fired(self):
        self._layer.governance_fired()

    def governance_skipped(self):
        self._layer.governance_skipped()

    def dense_fallback(self):
        self._layer.dense_fallback()

    def partial_repair(self):
        self._layer.partial_repair()

    def fusion_call(self):
        self._layer.fusion_call()

    def telemetry_suppressed(self):
        self._layer.telemetry_suppressed()

    def telemetry_emitted(self):
        self._layer.telemetry_emitted()

    def get_snapshot_json(self) -> str:
        return self._layer.get_snapshot_json()

    def get_stats(self) -> Dict[str, Any]:
        return json.loads(self.get_snapshot_json())

    @property
    def backend(self) -> str:
        return self._backend

    def reset_counters(self):
        self._layer.reset_counters()

    def format_live_line(self) -> str:
        s = self.get_stats()
        return (
            f"[NATIVE_TELEM/{self._backend}] "
            f"tokens={s.get('tokens_generated', 0)} "
            f"gov_skips={s.get('governance_skips', 0)} "
            f"skipped_pct={s.get('governance_collapse_ratio', 0.0):.1%} "
            f"telem_suppressed={s.get('telemetry_suppressed', 0)} "
            f"suppressed_pct={s.get('telemetry_suppress_ratio', 0.0):.1%}"
        )
