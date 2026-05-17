import os
import json
import time
from pathlib import Path
from typing import Dict, Any

class PrlTraceSystem:
    """
    STAGE 4A.2 — PRL: PRL Trace System.
    Writes strictly hardware-derived telemetry and latency events into discrete,
    line-by-line JSONL streams.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        
    def _get_file(self, trace_name: str):
        if trace_name not in self.files:
            file_path = self.output_dir / f"{trace_name}_trace.jsonl"
            self.files[trace_name] = open(file_path, "a", encoding="utf-8")
        return self.files[trace_name]
        
    def log_trace(self, trace_name: str, payload: Dict[str, Any]):
        """Persists trace record to disk synchronously."""
        if "timestamp" not in payload:
            payload["timestamp"] = time.time()
            
        payload["is_synthetic"] = False
        
        file = self._get_file(trace_name)
        file.write(json.dumps(payload) + "\n")
        file.flush()
        
    def close(self):
        """Closes all active trace file handles."""
        for f in self.files.values():
            try:
                f.close()
            except:
                pass
        self.files.clear()
