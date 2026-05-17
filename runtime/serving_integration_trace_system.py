"""
SIP Phase 41.2: Serving Integration Trace System.

Persists raw traces for Serving Integration Proof (SIP).

Trace files:
- execution_lineage_trace.jsonl
- stage_participation_trace.jsonl
- native_activation_trace.jsonl
- sparse_participation_trace.jsonl
- serving_path_trace.jsonl
- governance_activation_trace.jsonl

RAW traces only. No synthesis at write time.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any


class ServingIntegrationTraceSystem:
    """
    SIP Phase 41.2: Central trace persistence for execution lineage and participation proofs.
    """

    TRACE_FILES = {
        "execution_lineage":     "execution_lineage_trace.jsonl",
        "stage_participation":   "stage_participation_trace.jsonl",
        "native_activation":     "native_activation_trace.jsonl",
        "sparse_participation":  "sparse_participation_trace.jsonl",
        "serving_path":          "serving_path_trace.jsonl",
        "governance_activation": "governance_activation_trace.jsonl",
    }

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("SIP_TraceSystem")

        # Touch all trace files to ensure they exist
        for key, fname in self.TRACE_FILES.items():
            path = self.trace_dir / fname
            if not path.exists():
                path.touch()

        self._logger.info(
            "ServingIntegrationTraceSystem initialized | trace_dir=%s", self.trace_dir
        )

    # -----------------------------------------------------------------------
    # Typed write methods
    # -----------------------------------------------------------------------

    def write_execution_lineage(self, data: Dict[str, Any]):
        self._write("execution_lineage", data)

    def write_stage_participation(self, data: Dict[str, Any]):
        self._write("stage_participation", data)

    def write_native_activation(self, data: Dict[str, Any]):
        self._write("native_activation", data)

    def write_sparse_participation(self, data: Dict[str, Any]):
        self._write("sparse_participation", data)

    def write_serving_path(self, data: Dict[str, Any]):
        self._write("serving_path", data)

    def write_governance_activation(self, data: Dict[str, Any]):
        self._write("governance_activation", data)

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def get_trace_sizes(self) -> Dict[str, int]:
        sizes = {}
        for key, fname in self.TRACE_FILES.items():
            p = self.trace_dir / fname
            sizes[key] = p.stat().st_size if p.exists() else 0
        return sizes

    def get_trace_record_counts(self) -> Dict[str, int]:
        counts = {}
        for key, fname in self.TRACE_FILES.items():
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
        lines = ["[SIP TRACES]"]
        for key in self.TRACE_FILES:
            count = counts.get(key, 0)
            size = sizes.get(key, 0)
            lines.append("  %-25s %4d records  %6d bytes" % (key + ":", count, size))
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
