import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

class HumanUsageTraceSystem:
    """
    RHU Phase 40.3: Human Usage Trace System.
    Persists raw traces for human-facing operational coherence.
    """
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._init_traces()

    def _init_traces(self):
        trace_files = [
            "websocket_trace.jsonl",
            "session_continuity_trace.jsonl",
            "ux_stability_trace.jsonl",
            "browser_recovery_trace.jsonl",
            "human_interaction_trace.jsonl"
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

    def capture_ux_stability(self, session_id: str, score: float, jitter: float):
        self.log_trace("ux_stability", {"session_id": session_id, "score": score, "jitter": jitter})

    def log_browser_event(self, session_id: str, event_type: str):
        self.log_trace("browser_recovery", {"session_id": session_id, "event": event_type})
