import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

class InteractiveRuntimeTraceSystem:
    """
    OIS Phase 40.1: Interactive Runtime Trace System.
    Persists raw traces for session lifecycles, interruptions, and telemetry.
    """
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.loggers = {}
        self._init_traces()

    def _init_traces(self):
        trace_files = [
            "session_lifecycle_trace.jsonl",
            "streaming_trace.jsonl",
            "queue_recovery_trace.jsonl",
            "operational_failure_trace.jsonl",
            "live_telemetry_trace.jsonl"
        ]
        for tf in trace_files:
            # Ensure files exist
            (self.trace_dir / tf).touch(exist_ok=True)

    def log_trace(self, trace_type: str, data: Dict[str, Any]):
        """Logs a raw trace entry."""
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
            logging.error(f"Failed to write trace to {filename}: {e}")

    def capture_telemetry_snapshot(self, metrics: Dict[str, Any]):
        self.log_trace("live_telemetry", metrics)

    def log_session_event(self, session_id: str, event: str, metadata: Optional[Dict[str, Any]] = None):
        self.log_trace("session_lifecycle", {
            "session_id": session_id,
            "event": event,
            "metadata": metadata or {}
        })

    def log_operational_failure(self, failure_type: str, severity: str, details: str):
        self.log_trace("operational_failure", {
            "failure_type": failure_type,
            "severity": severity,
            "details": details
        })
