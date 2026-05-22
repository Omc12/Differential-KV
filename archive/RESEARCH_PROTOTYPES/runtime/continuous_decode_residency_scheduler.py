import os
import time
from pathlib import Path
from typing import Dict, List, Any

class ContinuousDecodeResidencyScheduler:
    """
    DPC Phase 42.1 — Continuous Decode Residency Scheduler.
    Maintains persistent slot resident states for ongoing generation batches,
    stabilizing queue pacing and avoiding tear-down/re-build overheads.
    """
    def __init__(self, workspace_root: Path, capacity: int = 8):
        self.workspace_root = workspace_root
        self.capacity = capacity
        self.active_slots: Dict[int, Dict[str, Any]] = {}
        
    def schedule_sequence(self, session_id: str, priority: int = 1) -> int:
        """Assigns session to a persistent active decode slot."""
        for slot in range(self.capacity):
            if slot not in self.active_slots:
                self.active_slots[slot] = {
                    "session_id": session_id,
                    "priority": priority,
                    "admitted_at": time.time(),
                    "active": True
                }
                print(f"[Residency Scheduler] Assigned slot {slot} to session {session_id}.")
                return slot
        raise RuntimeError("No resident slots available.")

    def release_slot(self, slot: int):
        if slot in self.active_slots:
            sid = self.active_slots[slot]["session_id"]
            del self.active_slots[slot]
            print(f"[Residency Scheduler] Released slot {slot} (session {sid}).")

    def get_residency_ratio(self) -> float:
        return len(self.active_slots) / self.capacity
