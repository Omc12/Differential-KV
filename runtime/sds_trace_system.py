import os
import json
from pathlib import Path
from typing import Dict, Any

class SdsTraceSystem:
    """
    SDS Trace System
    
    Streams and records exactly the 10 mandated hardware-derived SDS trace files.
    """
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_names = [
            "speculative_acceptance",
            "rollback",
            "verifier_alignment",
            "speculative_window",
            "replay_residency",
            "speculative_kv",
            "semantic_drift",
            "throughput_burst",
            "latency",
            "occupancy"
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
        self.handles.clear()
        
    def __del__(self):
        self.close()
