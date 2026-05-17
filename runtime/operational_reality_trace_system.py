import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

class OperationalRealityTraceSystem:
    """
    ORX Phase 40.2: Operational Reality Trace System.
    Persists raw traces for combined operational tracks.
    """
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._init_traces()

    def _init_traces(self):
        trace_files = [
            "long_session_trace.jsonl",
            "concurrency_trace.jsonl",
            "reconnect_trace.jsonl",
            "cancellation_trace.jsonl",
            "runtime_coherence_trace.jsonl"
        ]
        for tf in trace_files:
            (self.trace_dir / tf).touch(exist_ok=True)

    def log_trace(self, trace_type: str, data: Dict[str, Any]):
        filename = f"{trace_type}_trace.jsonl"
        path = self.trace_dir / filename
        
        entry = {
            "timestamp": time.time(),
            **data
        }
        
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logging.error(f"Trace persistence failed for {trace_type}: {e}")
            
    def capture_coherence(self, score: float, state: str):
        self.log_trace("runtime_coherence", {"score": score, "state": state})

    def log_concurrency_event(self, metrics: Dict[str, Any]):
        self.log_trace("concurrency", metrics)
