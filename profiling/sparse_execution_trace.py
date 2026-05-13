import json
import time
from typing import List, Dict, Any

class SparseExecutionTrace:
    """
    PHASE 7.5B: Sparse Execution Trace
    Generates structured JSON traces compatible with Chrome Tracing or Nsight,
    mapping high-level sparse operations to low-level GPU events.
    """
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log_event(self, name: str, phase: str, timestamp_us: int, tid: int = 0, args: Dict = None):
        """Adds a tracing event (B=begin, E=end)."""
        self.events.append({
            "name": name,
            "ph": phase,
            "ts": timestamp_us,
            "pid": 1,
            "tid": tid,
            "args": args or {}
        })

    def export_trace(self, filepath: str):
        """Saves the trace to a file."""
        with open(filepath, "w") as f:
            json.dump({"traceEvents": self.events}, f)

    def capture_step(self, step_name: str, duration_us: int):
        """Convenience method to log a complete step."""
        now = int(time.time() * 1000000)
        self.log_event(step_name, "B", now)
        self.log_event(step_name, "E", now + duration_us)
