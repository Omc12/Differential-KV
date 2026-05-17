import os
import json
from pathlib import Path
from typing import Dict, Any

class NcoTraceSystem:
    """
    NCO Trace System
    
    Manages and streams precisely the 10 mandated hardware-derived NCO trace files.
    """
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_names = [
            "continuous_serving",
            "adaptive_batch",
            "prefix_reuse",
            "stream_multiplex",
            "tail_latency",
            "speculative_decode",
            "queue_turbulence",
            "serving_continuity",
            "occupancy",
            "real_tps"
        ]
        
        self.handles = {}
        for name in self.trace_names:
            file_path = self.target_dir / f"{name}_trace.jsonl"
            self.handles[name] = open(file_path, "w", encoding="utf-8")

    def write_record(self, trace_name: str, record: Dict[str, Any]):
        """
        Writes a physical record to the designated JSONL trace file.
        """
        if trace_name in self.handles:
            self.handles[trace_name].write(json.dumps(record) + "\n")
            self.handles[trace_name].flush()

    def close(self):
        """
        Safely flushes and closes all open trace file handles.
        """
        for name, handle in self.handles.items():
            try:
                handle.close()
            except Exception:
                pass
