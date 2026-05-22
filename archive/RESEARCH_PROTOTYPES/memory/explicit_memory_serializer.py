import os
import json
import torch
from typing import Any, Dict

class ExplicitMemorySerializer:
    """
    Handles serialization and persistence of explicit memory structures.
    Ensures zero hidden-state leakage during save/load cycles.
    """
    def __init__(self, base_path: str = "./memory_checkpoints"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    def save_checkpoint(self, session_id: str, memory_state: Dict[str, Any]):
        """Save memory state to a JSON file."""
        file_path = os.path.join(self.base_path, f"{session_id}.json")
        with open(file_path, "w") as f:
            json.dump(memory_state, f, indent=2)
            
    def load_checkpoint(self, session_id: str) -> Dict[str, Any]:
        """Load memory state from a JSON file."""
        file_path = os.path.join(self.base_path, f"{session_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No checkpoint found for session {session_id}")
        with open(file_path, "r") as f:
            return json.load(f)

    def verify_no_tensors(self, state: Dict[str, Any]):
        """
        Adversarial auditor: ensures no torch.Tensors or raw activations 
        are accidentally serialized.
        """
        def _check(obj):
            if isinstance(obj, torch.Tensor):
                raise ValueError("Hidden tensor detected in explicit memory!")
            if isinstance(obj, dict):
                for v in obj.values():
                    _check(v)
            elif isinstance(obj, list):
                for item in obj:
                    _check(item)
        _check(state)
        return True
