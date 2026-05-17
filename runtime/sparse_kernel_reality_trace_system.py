"""
SKO Phase 41.3: Sparse Kernel Reality Trace System.

Persists raw traces for Sparse Kernel Optimization (SKO).

Trace files:
- sparse_kernel_occupancy_trace.jsonl
- sparse_memory_locality_trace.jsonl
- sparse_pipeline_fusion_trace.jsonl
- sparse_attention_fusion_trace.jsonl
- sparse_gpu_metadata_trace.jsonl
- sparse_kernel_stall_trace.jsonl
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any


class SparseKernelRealityTraceSystem:
    TRACE_FILES = {
        "sparse_kernel_occupancy": "sparse_kernel_occupancy_trace.jsonl",
        "sparse_memory_locality": "sparse_memory_locality_trace.jsonl",
        "sparse_pipeline_fusion": "sparse_pipeline_fusion_trace.jsonl",
        "sparse_attention_fusion": "sparse_attention_fusion_trace.jsonl",
        "sparse_gpu_metadata": "sparse_gpu_metadata_trace.jsonl",
        "sparse_kernel_stall": "sparse_kernel_stall_trace.jsonl",
    }

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("SKO_TraceSystem")

        for key, fname in self.TRACE_FILES.items():
            path = self.trace_dir / fname
            if not path.exists():
                path.touch()

    def write_occupancy(self, data: Dict[str, Any]):
        self._write("sparse_kernel_occupancy", data)

    def write_locality(self, data: Dict[str, Any]):
        self._write("sparse_memory_locality", data)

    def write_pipeline_fusion(self, data: Dict[str, Any]):
        self._write("sparse_pipeline_fusion", data)

    def write_attention_fusion(self, data: Dict[str, Any]):
        self._write("sparse_attention_fusion", data)

    def write_gpu_metadata(self, data: Dict[str, Any]):
        self._write("sparse_gpu_metadata", data)

    def write_kernel_stall(self, data: Dict[str, Any]):
        self._write("sparse_kernel_stall", data)

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
