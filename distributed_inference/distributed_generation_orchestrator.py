import torch
from typing import Dict, List, Any, Optional
import logging

class DistributedGenerationOrchestrator:
    """
    Coordinates distributed token generation across multiple devices.
    Manages the autoregressive loop and sparse decode orchestration.
    """
    def __init__(self, devices: List[str]):
        self.devices = devices
        self.generation_history: List[int] = []
        self.active_sessions: Dict[str, Any] = {}
        self.logger = logging.getLogger("DistributedGenerationOrchestrator")

    def start_generation(self, session_id: str, prompt_ids: List[int]):
        """Initializes a distributed generation session."""
        self.active_sessions[session_id] = {
            "token_ids": prompt_ids.copy(),
            "step": 0,
            "status": "running"
        }
        self.logger.info(f"Started generation session {session_id} with {len(prompt_ids)} tokens.")

    def step_generation(self, session_id: str, next_token: int, device: str):
        """Records a generated token and updates session state."""
        if session_id not in self.active_sessions:
            raise KeyError(f"Session {session_id} not found.")
        
        session = self.active_sessions[session_id]
        session["token_ids"].append(next_token)
        session["step"] += 1
        
        self.logger.info(f"Step {session['step']} for {session_id}: generated {next_token} on {device}")

    def get_session_tokens(self, session_id: str) -> List[int]:
        return self.active_sessions[session_id]["token_ids"]
