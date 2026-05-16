"""
STAGE 2 — SAT: Transformer Execution Mode Trace
Phase 38.9 — Sparse Attention Transition

Persists per-layer, per-step execution metadata:
  - sparse mode / dense mode / hybrid mode
  - fallback reason (when not sparse)
  - wall-clock execution duration

RAW TRACES ONLY — no interpretation, no thresholding.
"""

import time
import json
import threading
import os
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any


TRACE_PATH = "traces/stage2/phase_38_9_sat/execution_mode_trace.jsonl"


@dataclass
class LayerStepRecord:
    step: int
    layer_idx: int
    mode: str
    fallback_reason: Optional[str]
    duration_ms: float
    token_count: int
    ts: float = field(default_factory=time.time)


class TransformerExecutionModeTrace:
    """
    STAGE 2 SAT: Transformer Execution Mode Trace.

    Records per-layer execution metadata for every transformer decode step.

    Usage:
        tracer = TransformerExecutionModeTrace()
        tracer.record(step=0, layer_idx=3, mode="sparse", duration_ms=1.4, token_count=512)
        summary = tracer.get_summary()
        tracer.flush_and_close()
    """

    def __init__(self, trace_path: str = TRACE_PATH, flush_every: int = 64):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

        self._lock = threading.Lock()
        self._flush_every = flush_every
        self._trace_buf: List[Dict] = []

        self._mode_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._mode_ms: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._fallback_reasons: Dict[str, int] = defaultdict(int)
        self._step_mode: Dict[int, Dict[str, int]] = {}

        self._total_records = 0
        self._session_start = time.time()

    def record(
        self,
        step: int,
        layer_idx: int,
        mode: str,
        duration_ms: float,
        token_count: int = 0,
        fallback_reason: Optional[str] = None,
    ) -> None:
        ts = time.time()
        with self._lock:
            self._total_records += 1
            lk = str(layer_idx)
            self._mode_counts[lk][mode] += 1
            self._mode_ms[lk][mode] += duration_ms
            if fallback_reason:
                self._fallback_reasons[fallback_reason] += 1
            if step not in self._step_mode:
                self._step_mode[step] = defaultdict(int)
            self._step_mode[step][mode] += 1

            entry = asdict(LayerStepRecord(
                step=step, layer_idx=layer_idx, mode=mode,
                fallback_reason=fallback_reason,
                duration_ms=round(duration_ms, 4),
                token_count=token_count, ts=ts,
            ))
            self._trace_buf.append(entry)
            if len(self._trace_buf) >= self._flush_every:
                self._flush()

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self._session_start
            global_counts: Dict[str, int] = defaultdict(int)
            global_ms: Dict[str, float] = defaultdict(float)
            for lk, modes in self._mode_counts.items():
                for m, cnt in modes.items():
                    global_counts[m] += cnt
                    global_ms[m] += self._mode_ms[lk].get(m, 0.0)
            total = max(sum(global_counts.values()), 1)
            return {
                "total_records": self._total_records,
                "elapsed_sec": round(elapsed, 2),
                "mode_distribution": {
                    m: {
                        "count": cnt,
                        "fraction": round(cnt / total, 4),
                        "total_ms": round(global_ms[m], 4),
                        "avg_ms": round(global_ms[m] / max(cnt, 1), 4),
                    }
                    for m, cnt in global_counts.items()
                },
                "fallback_reasons": dict(self._fallback_reasons),
                "steps_recorded": len(self._step_mode),
            }

    def get_layer_breakdown(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for lk in sorted(self._mode_counts.keys(), key=lambda x: int(x)):
                modes = self._mode_counts[lk]
                ms_map = self._mode_ms[lk]
                total = max(sum(modes.values()), 1)
                rows.append({
                    "layer": int(lk),
                    "total_calls": total,
                    "sparse_calls": modes.get("sparse", 0),
                    "dense_calls": modes.get("dense", 0),
                    "hybrid_calls": modes.get("hybrid", 0),
                    "sparse_fraction": round(modes.get("sparse", 0) / total, 4),
                    "dense_fraction": round(modes.get("dense", 0) / total, 4),
                    "avg_sparse_ms": round(ms_map.get("sparse", 0) / max(modes.get("sparse", 1), 1), 4),
                    "avg_dense_ms": round(ms_map.get("dense", 0) / max(modes.get("dense", 1), 1), 4),
                })
            return rows

    def get_dominant_mode(self) -> str:
        with self._lock:
            global_counts: Dict[str, int] = defaultdict(int)
            for modes in self._mode_counts.values():
                for m, cnt in modes.items():
                    global_counts[m] += cnt
            if not global_counts:
                return "unknown"
            return max(global_counts, key=lambda k: global_counts[k])

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        if not self._trace_buf:
            return
        with open(self.trace_path, "a") as f:
            for ev in self._trace_buf:
                f.write(json.dumps(ev) + "\n")
        self._trace_buf.clear()
