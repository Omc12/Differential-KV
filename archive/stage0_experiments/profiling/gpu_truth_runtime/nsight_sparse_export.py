import torch
import json
import os

class NsightSparseExport:
    """
    Exports sparse-runtime traces in a format compatible with Nsight profiling.
    Allows deep-dive hardware analysis of retrieval kernels.
    """
    def __init__(self, export_path: str = "profiling/traces/"):
        self.export_path = export_path
        os.makedirs(export_path, exist_ok=True)
        self.trace_data = []

    def push_range(self, message: str, color: str = "blue"):
        """Pushes a Nsight NVTX range."""
        try:
            torch.cuda.nvtx.range_push(message)
        except:
            pass # NVTX might not be available

    def pop_range(self):
        """Pops a Nsight NVTX range."""
        try:
            torch.cuda.nvtx.range_pop()
        except:
            pass

    def record_trace(self, kernel_name: str, start_ts: float, end_ts: float, metadata: dict):
        """Records a trace event for offline export."""
        self.trace_data.append({
            "name": kernel_name,
            "cat": "sparse_runtime",
            "ph": "X",
            "ts": start_ts * 1000000, # Convert to microseconds
            "dur": (end_ts - start_ts) * 1000000,
            "args": metadata
        })

    def export_chrome_trace(self, filename: str = "sparse_trace.json"):
        """Exports the trace in Chrome Trace Format (viewable in Nsight/chrome://tracing)."""
        full_path = os.path.join(self.export_path, filename)
        with open(full_path, "w") as f:
            json.dump(self.trace_data, f)
        return full_path
