import os
import json
from pathlib import Path
from typing import Dict, Any

class SspTraceSystem:
    """
    SSP Trace System
    
    Streams precisely the 10 mandated JSONL traces under stage4b/phase_45_2_ssp.
    """
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # File handles for the 10 mandated JSONL traces
        self.trace_names = [
            "semantic_continuity",
            "weak_signal",
            "planning",
            "abstraction",
            "synthesis",
            "extractive_collapse",
            "discourse",
            "semantic_drift",
            "semantic_blending",
            "ollama_semantic_comparison"
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
