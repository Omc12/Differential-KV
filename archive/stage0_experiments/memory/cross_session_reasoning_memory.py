"""
memory/cross_session_reasoning_memory.py

Enables cross-session reasoning continuity by serializing cognitive fields.
"""

import torch
import os
import json
from typing import Dict, Any, Tuple

class CrossSessionReasoningMemory:
    """
    Saves and loads cognitive states to/from disk.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def save_session(self, session_id: str, cognitive_field: torch.Tensor, metadata: Dict[str, Any]):
        """
        Saves the cognitive field and metadata.
        """
        torch.save(cognitive_field, os.path.join(self.storage_path, f"{session_id}_field.pt"))
        with open(os.path.join(self.storage_path, f"{session_id}_meta.json"), "w") as f:
            json.dump(metadata, f)

    def load_session(self, session_id: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Loads a previously saved session.
        """
        field = torch.load(os.path.join(self.storage_path, f"{session_id}_field.pt"))
        with open(os.path.join(self.storage_path, f"{session_id}_meta.json"), "r") as f:
            metadata = json.load(f)
        return field, metadata

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        csrm = CrossSessionReasoningMemory(tmpdir)
        
        field = torch.randn(8, 128, 64)
        meta = {"task": "recursive_planning", "step": 42}
        
        csrm.save_session("session_001", field, meta)
        f_loaded, m_loaded = csrm.load_session("session_001")
        
        print(f"Session Saved and Loaded Successfully.")
        print(f"Metadata Match: {meta == m_loaded}")
        print(f"Field Shape: {f_loaded.shape}")
