"""
PCR Phase 41.4.5: Physical Compute Trace System.
Persists raw traces for Physical Compute Reality (PCR).
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

class PhysicalComputeTraceSystem:
    TRACE_FILES = {
        "cuda_kernel": "cuda_kernel_trace.jsonl",
        "transformer_compute": "transformer_compute_trace.jsonl",
        "gpu_load": "gpu_load_trace.jsonl",
        "dense_sparse_comparison": "dense_sparse_comparison_trace.jsonl",
        "context_scaling": "context_scaling_trace.jsonl",
        "gpu_timeline": "gpu_timeline_trace.jsonl",
    }

    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("PCR_TraceSystem")

        for key, fname in self.TRACE_FILES.items():
            path = self.trace_dir / fname
            if not path.exists():
                path.touch()

    def write_cuda_kernel(self, data: Dict[str, Any]):
        self._write("cuda_kernel", data)

    def write_transformer_compute(self, data: Dict[str, Any]):
        self._write("transformer_compute", data)

    def write_gpu_load(self, data: Dict[str, Any]):
        self._write("gpu_load", data)

    def write_dense_sparse_comparison(self, data: Dict[str, Any]):
        self._write("dense_sparse_comparison", data)

    def write_context_scaling(self, data: Dict[str, Any]):
        self._write("context_scaling", data)

    def write_gpu_timeline(self, data: Dict[str, Any]):
        self._write("gpu_timeline", data)

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
