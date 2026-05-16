import time
import json
import os
from typing import Dict, List, Any
import logging

class DecodeOverlapTelemetry:
    """
    STAGE 2 CDBE: Decode Overlap Telemetry.
    Tracks REAL overlapping decode counts and occupancy continuity.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.logger = logging.getLogger("CDBETelemetry")
        self.history = []
        
        self.active_sessions = set()
        self.batch_sizes = []
        self.step_durations = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def record_admission(self, session_id: str):
        self.active_sessions.add(session_id)
        self._log_event("ADMISSION", {"session_id": session_id, "active_count": len(self.active_sessions)})

    def record_completion(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
            self._log_event("COMPLETION", {"session_id": session_id, "active_count": len(self.active_sessions)})

    def record_decode_step(self, batch_size: int, duration_ms: float):
        self.batch_sizes.append(batch_size)
        self.step_durations.append(duration_ms)
        
        if len(self.batch_sizes) % 10 == 0:
            avg_batch = sum(self.batch_sizes[-10:]) / 10.0
            avg_duration = sum(self.step_durations[-10:]) / 10.0
            self._log_event("DECODE_WINDOW", {
                "overlap_count": len(self.active_sessions),
                "avg_batch_size": avg_batch,
                "step_ms": avg_duration,
                "occupancy_continuity": avg_batch / max(1, len(self.active_sessions))
            })

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "event": event_type,
            **data
        }
        self.history.append(entry)
        
        # Append to file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_summary(self) -> Dict[str, Any]:
        if not self.batch_sizes:
            return {"status": "NO_DATA"}
            
        return {
            "total_decode_steps": len(self.batch_sizes),
            "max_overlap": max([h["active_count"] for h in self.history if "active_count" in h] or [0]),
            "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes),
            "avg_step_duration_ms": sum(self.step_durations) / len(self.step_durations),
            "occupancy_continuity_score": (sum(self.batch_sizes) / len(self.batch_sizes)) / max(1, max(self.batch_sizes)) if self.batch_sizes else 0
        }
