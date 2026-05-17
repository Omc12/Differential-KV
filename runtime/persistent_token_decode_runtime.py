import os
import time
from pathlib import Path
from typing import Dict, List, Any

class PersistentTokenDecodeRuntime:
    """
    DPC Phase 42.1 — Persistent Token Decode Runtime.
    Manages continuous, rolling token executions and persistent decode states
    across generation steps to avoid per-token model state rebuilds.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.decode_states: Dict[str, Dict[str, Any]] = {}
        self.active_generation = False

    def init_decode_state(self, session_id: str, initial_state: Dict[str, Any]):
        """Registers a persistent decode state for rolling execution."""
        self.decode_states[session_id] = {
            "session_id": session_id,
            "step": 0,
            "kv_cache_refs": initial_state.get("kv_cache_refs", {}),
            "state_tensors": initial_state.get("state_tensors", {}),
            "last_active": time.time()
        }
        self.active_generation = True
        print(f"[Persistent Decode] State registered for session {session_id}.")

    def reuse_and_advance(self, session_id: str) -> Dict[str, Any]:
        """Retrieves the persistent decode state and increments step."""
        if session_id not in self.decode_states:
            raise KeyError(f"No persistent state for session {session_id}")
            
        state = self.decode_states[session_id]
        state["step"] += 1
        state["last_active"] = time.time()
        return state

    def teardown_state(self, session_id: str):
        if session_id in self.decode_states:
            del self.decode_states[session_id]
            print(f"[Persistent Decode] State cleared for session {session_id}.")
        if not self.decode_states:
            self.active_generation = False
