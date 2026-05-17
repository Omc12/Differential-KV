"""
PRD Phase 41.0: Performance Reality Trace System.
Persists raw runtime traces from all PRD profiling systems.

Persists:
- runtime timing traces
- governance cost traces
- queue turbulence traces
- GPU occupancy traces
- dense fallback traces
- control-plane traces

RAW traces only. No synthesis, no aggregation at write time.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class PerformanceRealityTraceSystem:
    """
    PRD Phase 41.0: Central trace persistence hub for all PRD profiling data.
    Provides a single write path to prevent file descriptor contention.
    All subsystems delegate trace I/O to this class.
    """

    TRACE_FILES = {
        "runtime_timing":   "runtime_timing_trace.jsonl",
        "governance_cost":  "governance_cost_trace.jsonl",
        "gpu_occupancy":    "gpu_occupancy_trace.jsonl",
        "dense_fallback":   "dense_fallback_trace.jsonl",
        "control_plane":    "control_plane_trace.jsonl",
        "queue_turbulence": "queue_turbulence_trace.jsonl",
        "raw_nvidia_smi":   "raw_nvidia_smi_dmon.log",  # Written by GPU analyzer
    }

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("PRD_TraceSystem")

        # Touch all trace files to ensure they exist for validation
        for key, fname in self.TRACE_FILES.items():
            if not fname.endswith(".log"):  # Don't touch raw log
                path = self.trace_dir / fname
                if not path.exists():
                    path.touch()

        self._logger.info(f"PerformanceRealityTraceSystem initialized → {self.trace_dir}")
        self._logger.info(f"Active trace files: {list(self.TRACE_FILES.values())}")

    # -----------------------------------------------------------------------
    # Typed write methods — one per trace type
    # -----------------------------------------------------------------------

    def write_runtime_timing(self, data: Dict[str, Any]):
        """Persist a runtime timing record."""
        self._write("runtime_timing", data)

    def write_governance_cost(self, data: Dict[str, Any]):
        """Persist a governance cost decomposition record."""
        self._write("governance_cost", data)

    def write_gpu_occupancy(self, data: Dict[str, Any]):
        """Persist a GPU occupancy sample."""
        self._write("gpu_occupancy", data)

    def write_dense_fallback(self, data: Dict[str, Any]):
        """Persist a dense fallback event."""
        self._write("dense_fallback", data)

    def write_control_plane(self, data: Dict[str, Any]):
        """Persist a control-plane weight snapshot."""
        self._write("control_plane", data)

    def write_queue_turbulence(self, data: Dict[str, Any]):
        """Persist a queue/scheduler event."""
        self._write("queue_turbulence", data)

    # -----------------------------------------------------------------------
    # Generic write
    # -----------------------------------------------------------------------

    def write(self, trace_type: str, data: Dict[str, Any]):
        """Generic write by trace type key."""
        self._write(trace_type, data)

    # -----------------------------------------------------------------------
    # Inspection utilities
    # -----------------------------------------------------------------------

    def get_trace_sizes(self) -> Dict[str, int]:
        """Return byte size of each trace file."""
        sizes = {}
        for key, fname in self.TRACE_FILES.items():
            path = self.trace_dir / fname
            sizes[key] = path.stat().st_size if path.exists() else 0
        return sizes

    def get_trace_record_counts(self) -> Dict[str, int]:
        """Count JSONL records in each trace file."""
        counts = {}
        for key, fname in self.TRACE_FILES.items():
            path = self.trace_dir / fname
            if not path.exists() or fname.endswith(".log"):
                counts[key] = 0
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    counts[key] = sum(1 for line in f if line.strip())
            except Exception:
                counts[key] = -1
        return counts

    def status_summary(self) -> str:
        counts = self.get_trace_record_counts()
        sizes = self.get_trace_sizes()
        lines = ["[PRD TRACES]"]
        for key in self.TRACE_FILES:
            count = counts.get(key, 0)
            size = sizes.get(key, 0)
            lines.append(f"  {key}: {count} records, {size} bytes")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Internal write primitive
    # -----------------------------------------------------------------------

    def _write(self, trace_type: str, data: Dict[str, Any]):
        fname = self.TRACE_FILES.get(trace_type)
        if not fname:
            self._logger.warning(f"Unknown trace type: {trace_type}")
            return
        if fname.endswith(".log"):
            return  # Raw log is written by GPU analyzer subprocess

        record = {"timestamp": time.time(), **data}
        path = self.trace_dir / fname
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error(f"Trace write error [{trace_type}]: {e}")
