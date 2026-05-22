import os
import time
from pathlib import Path
from typing import Dict, List, Any

class ContinuousBatchResidencyEngine:
    """
    CGO Phase 42.0 — Continuous Batch Residency Engine.
    Manages persistent decode slots, schedules rolling admissions, and prevents
    frequent batch teardown cycles to keep the GPU continuously occupied.
    """
    def __init__(self, workspace_root: Path, max_slots: int = 16):
        self.workspace_root = workspace_root
        self.max_slots = max_slots
        self.active_slots: Dict[int, Optional[str]] = {i: None for i in range(max_slots)}
        self.admitted_times: Dict[str, float] = {}
        self.session_sequences: Dict[str, List[int]] = {}

    def admit_session(self, session_id: str, prompt_ids: List[int]) -> bool:
        """
        Admits a new session into an empty persistent decode slot.
        Avoids batch rebuild overhead.
        """
        for slot, active_sid in self.active_slots.items():
            if active_sid is None:
                self.active_slots[slot] = session_id
                self.admitted_times[session_id] = time.time()
                self.session_sequences[session_id] = list(prompt_ids)
                print(f"[Batch Residency] Session {session_id} admitted into slot {slot}.")
                return True
        return False # No free slot

    def evict_session(self, session_id: str):
        """
        Evicts a completed session from its slot, freeing it for rolling admission.
        """
        for slot, active_sid in self.active_slots.items():
            if active_sid == session_id:
                self.active_slots[slot] = None
                if session_id in self.admitted_times:
                    del self.admitted_times[session_id]
                print(f"[Batch Residency] Session {session_id} evicted from slot {slot}.")
                return

    def get_occupancy_rate(self) -> float:
        """Calculates current slots residency occupancy."""
        occupied = sum(1 for sid in self.active_slots.values() if sid is not None)
        return occupied / self.max_slots

    def step_batch(self, generated_tokens: Dict[str, int]):
        """
        Appends generated tokens to the active session sequences.
        """
        for sid, tok in generated_tokens.items():
            if sid in self.session_sequences:
                self.session_sequences[sid].append(tok)
