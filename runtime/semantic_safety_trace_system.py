import json
import time
from pathlib import Path
from typing import Dict, Any

class SemanticSafetyTraceSystem:
    """
    STAGE 2 - SRI: Semantic Safety Trace System
    Persists RAW traces for semantic repair integration validation.
    """
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self.traces = {
            "anchor_integrity": self.trace_dir / "anchor_integrity_trace.jsonl",
            "semantic_repair": self.trace_dir / "semantic_repair_trace.jsonl",
            "semantic_safety": self.trace_dir / "semantic_safety_trace.jsonl",
            "fallback_recovery": self.trace_dir / "fallback_recovery_trace.jsonl",
            "sparse_correctness": self.trace_dir / "sparse_correctness_trace.jsonl"
        }

    def log_event(self, trace_key: str, data: Dict[str, Any]):
        if trace_key not in self.traces:
            return
            
        payload = {
            "ts": time.time(),
            **data
        }
        
        with open(self.traces[trace_key], "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
