import json
import time

class SparseExecutionTrace:
    """
    Generates a high-fidelity trace of sparse execution events.
    Compatible with Chrome Tracing format for visualization.
    """
    def __init__(self):
        self.events = []

    def log_event(self, name: str, cat: str, ph: str, args: dict = None):
        """
        ph: 'B' (Begin), 'E' (End), 'X' (Complete)
        """
        self.events.append({
            "name": name,
            "cat": cat,
            "ph": ph,
            "ts": time.perf_counter() * 1e6, # Microseconds
            "pid": 1,
            "tid": 1,
            "args": args or {}
        })

    def save_trace(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.events, f)
