"""
RCO-N Phase 41.1: Runtime Optimization Trace System.

Persists raw optimization traces from all RCO-N subsystems.

Trace files:
- gpu_idle_gap_trace.jsonl
- orchestration_collapse_trace.jsonl
- persistent_batch_trace.jsonl
- partial_dense_recovery_trace.jsonl
- scheduler_fragmentation_trace.jsonl
- native_scheduler_trace.jsonl
- native_sparse_metadata_trace.jsonl
- queue_turbulence_collapse_trace.jsonl
- gpu_saturation_trace.jsonl

RAW traces only. No synthesis at write time.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any


class RuntimeOptimizationTraceSystem:
    """
    RCO-N Phase 41.1: Central trace persistence for all optimization subsystems.
    Mirrors the PRD PerformanceRealityTraceSystem pattern.
    """

    TRACE_FILES = {
        "gpu_idle_gap":              "gpu_idle_gap_trace.jsonl",
        "orchestration_collapse":    "orchestration_collapse_trace.jsonl",
        "persistent_batch":          "persistent_batch_trace.jsonl",
        "partial_dense_recovery":    "partial_dense_recovery_trace.jsonl",
        "scheduler_fragmentation":   "scheduler_fragmentation_trace.jsonl",
        "native_scheduler":          "native_scheduler_trace.jsonl",
        "native_sparse_metadata":    "native_sparse_metadata_trace.jsonl",
        "queue_turbulence_collapse": "queue_turbulence_collapse_trace.jsonl",
        "gpu_saturation":            "gpu_saturation_trace.jsonl",
        "runtime_timing":            "runtime_timing_trace.jsonl",
        "gpu_occupancy":             "gpu_occupancy_trace.jsonl",
        "raw_nvidia_smi":            None,  # Written by GPU analyzer directly
    }

    TELEMETRY_FILES = {
        "raw_nvidia_smi_dmon": "raw_nvidia_smi_dmon.log",
    }

    def __init__(self, trace_dir: Path, telemetry_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.telemetry_dir = Path(telemetry_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("RCO_TraceSystem")

        # Touch all trace files to ensure they exist
        for key, fname in self.TRACE_FILES.items():
            if fname is not None:
                path = self.trace_dir / fname
                if not path.exists():
                    path.touch()

        # Touch telemetry files
        for key, fname in self.TELEMETRY_FILES.items():
            path = self.telemetry_dir / fname
            if not path.exists():
                path.touch()

        self._logger.info(
            "RuntimeOptimizationTraceSystem initialized | "
            "trace_dir=%s | telemetry_dir=%s",
            self.trace_dir, self.telemetry_dir
        )

    # -----------------------------------------------------------------------
    # Typed write methods
    # -----------------------------------------------------------------------

    def write_gpu_idle_gap(self, data: Dict[str, Any]):
        self._write("gpu_idle_gap", data)

    def write_orchestration_collapse(self, data: Dict[str, Any]):
        self._write("orchestration_collapse", data)

    def write_persistent_batch(self, data: Dict[str, Any]):
        self._write("persistent_batch", data)

    def write_partial_dense_recovery(self, data: Dict[str, Any]):
        self._write("partial_dense_recovery", data)

    def write_scheduler_fragmentation(self, data: Dict[str, Any]):
        self._write("scheduler_fragmentation", data)

    def write_native_scheduler(self, data: Dict[str, Any]):
        self._write("native_scheduler", data)

    def write_native_sparse_metadata(self, data: Dict[str, Any]):
        self._write("native_sparse_metadata", data)

    def write_queue_turbulence_collapse(self, data: Dict[str, Any]):
        self._write("queue_turbulence_collapse", data)

    def write_gpu_saturation(self, data: Dict[str, Any]):
        self._write("gpu_saturation", data)

    def write_runtime_timing(self, data: Dict[str, Any]):
        self._write("runtime_timing", data)

    def write_gpu_occupancy(self, data: Dict[str, Any]):
        self._write("gpu_occupancy", data)

    def write(self, trace_type: str, data: Dict[str, Any]):
        self._write(trace_type, data)

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def get_trace_sizes(self) -> Dict[str, int]:
        sizes = {}
        for key, fname in self.TRACE_FILES.items():
            if fname is None:
                sizes[key] = 0
                continue
            p = self.trace_dir / fname
            sizes[key] = p.stat().st_size if p.exists() else 0
        return sizes

    def get_trace_record_counts(self) -> Dict[str, int]:
        counts = {}
        for key, fname in self.TRACE_FILES.items():
            if fname is None:
                counts[key] = 0
                continue
            p = self.trace_dir / fname
            if not p.exists():
                counts[key] = 0
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    counts[key] = sum(1 for ln in f if ln.strip())
            except Exception:
                counts[key] = -1
        return counts

    def status_summary(self) -> str:
        counts = self.get_trace_record_counts()
        sizes = self.get_trace_sizes()
        lines = ["[RCO-N TRACES]"]
        for key in self.TRACE_FILES:
            count = counts.get(key, 0)
            size = sizes.get(key, 0)
            lines.append("  %-35s %4d records  %6d bytes" % (key + ":", count, size))
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Internal write primitive
    # -----------------------------------------------------------------------

    def _write(self, trace_type: str, data: Dict[str, Any]):
        fname = self.TRACE_FILES.get(trace_type)
        if fname is None:
            return
        record = {"timestamp": time.time(), **data}
        path = self.trace_dir / fname
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error("Trace write error [%s]: %s", trace_type, e)
