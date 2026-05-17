import time
import json
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class ContinuousDynamicBatchRuntime:
    """
    SGC Stage 3C.4: Continuous Dynamic Batch Runtime.
    Manages a persistent, rolling active decode window where requests are
    admitted and scheduled continuously, preventing stop-start decode cycles.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Continuous batching metrics
        self.batch_continuity = 100.0  # continuous batch residency %
        self.rolling_occupancy = 0.0   # average active session occupancy
        self.starvation_events = 0     # GPU starvation intervals detected
        self.last_step_time = time.perf_counter()

    def admit_request(self, session_id: str, prompt: str, max_new_tokens: int, input_ids: torch.Tensor):
        """
        Admits a new request rolling into the active decode pool.
        """
        self.active_sessions[session_id] = {
            "prompt": prompt,
            "input_ids": input_ids.clone(),
            "generated_tokens": [],
            "max_new_tokens": max_new_tokens,
            "status": "prefill",
            "admitted_at": time.time(),
            "last_active": time.time()
        }
        self._update_occupancy()

    def get_active_batch(self) -> List[str]:
        """
        Retrieves all currently active session IDs scheduled in the batch.
        """
        return [sid for sid, s in self.active_sessions.items() if s["status"] in ["prefill", "decode"]]

    def step_completed(self, session_id: str, next_token: int):
        """
        Records completion of a single token decode step for a session.
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["generated_tokens"].append(next_token)
            session["last_active"] = time.time()
            
            # Update status
            if len(session["generated_tokens"]) >= session["max_new_tokens"]:
                session["status"] = "completed"
            else:
                session["status"] = "decode"
                
        self._update_occupancy()

    def detect_starvation(self) -> bool:
        """
        Checks if the GPU is starved because the scheduling queue is empty.
        """
        now = time.perf_counter()
        idle_duration = now - self.last_step_time
        
        # If there are no active sessions and we wait more than 50ms, record starvation
        if len(self.get_active_batch()) == 0 and idle_duration > 0.05:
            self.starvation_events += 1
            self.batch_continuity = max(0.0, self.batch_continuity - 5.0)
            self.last_step_time = now
            return True
            
        self.batch_continuity = min(100.0, self.batch_continuity + 1.0)
        self.last_step_time = now
        return False

    def _update_occupancy(self):
        """
        Updates average active batch occupancy ratio.
        """
        active_count = len(self.get_active_batch())
        # Normalize occupancy to a scale out of 8 maximum concurrent batch slots
        self.rolling_occupancy = min(100.0, (active_count / 8.0) * 100.0)

    def clear(self):
        """
        Resets batch states.
        """
        self.active_sessions.clear()
        self.batch_continuity = 100.0
        self.rolling_occupancy = 0.0
        self.starvation_events = 0
