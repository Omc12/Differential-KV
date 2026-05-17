"""
MRO Phase 41.4: Memory Realization Trace System.
Persists raw traces for Memory Realization Optimization (MRO).
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

class MemoryRealizationTraceSystem:
    TRACE_FILES = {
        "sparse_kv_compaction": "sparse_kv_compaction_trace.jsonl",
        "sparse_residency": "sparse_residency_trace.jsonl",
        "vram_fragmentation": "vram_fragmentation_trace.jsonl",
        "long_context_memory": "long_context_memory_trace.jsonl",
        "multi_session_memory": "multi_session_memory_trace.jsonl",
        "residency_prediction": "residency_prediction_trace.jsonl",
    }

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("MRO_TraceSystem")

        for key, fname in self.TRACE_FILES.items():
            path = self.trace_dir / fname
            if not path.exists():
                path.touch()

    def write_compaction(self, data: Dict[str, Any]):
        self._write("sparse_kv_compaction", data)

    def write_residency(self, data: Dict[str, Any]):
        self._write("sparse_residency", data)

    def write_fragmentation(self, data: Dict[str, Any]):
        self._write("vram_fragmentation", data)

    def write_long_context(self, data: Dict[str, Any]):
        self._write("long_context_memory", data)

    def write_multi_session(self, data: Dict[str, Any]):
        self._write("multi_session_memory", data)

    def write_prediction(self, data: Dict[str, Any]):
        self._write("residency_prediction", data)

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

    def _write(self, trace_type: str, data: Dict[str, Any]):
        fname = self.TRACE_FILES.get(trace_type)
        if not fname: return
        path = self.trace_dir / fname
        try:
            with open(path, "a", encoding="utf-8") as f:
                record = {"timestamp": time.time(), **data}
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.error("Failed writing trace %s: %s", trace_type, e)
