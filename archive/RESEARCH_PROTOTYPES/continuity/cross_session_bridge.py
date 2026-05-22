import torch
import json
import os
from typing import Dict, Any, Optional, List

class CrossSessionBridge:
    """
    Serializes and restores cognitive state across different execution sessions.
    """
    def __init__(self, storage_dir: str = "session_checkpoints"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def checkpoint_session(self, session_id: str, state: Dict[str, Any]):
        """
        Saves the current cognitive state to a checkpoint.
        """
        path = os.path.join(self.storage_dir, f"session_{session_id}.pt")
        
        # We handle both torch tensors and JSON-serializable metadata
        torch_data = {}
        metadata = {}
        
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                torch_data[k] = v
            else:
                metadata[k] = v
                
        torch.save({
            "tensors": torch_data,
            "metadata": metadata
        }, path)
        print(f"Session {session_id} checkpointed successfully.")

    def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Restores a cognitive state from a checkpoint.
        """
        path = os.path.join(self.storage_dir, f"session_{session_id}.pt")
        if not os.path.exists(path):
            print(f"Checkpoint for session {session_id} not found.")
            return None
            
        data = torch.load(path)
        state = {**data["tensors"], **data["metadata"]}
        print(f"Session {session_id} restored successfully.")
        return state

    def list_checkpoints(self) -> List[str]:
        """
        Lists all available session checkpoints.
        """
        return [f.replace("session_", "").replace(".pt", "") 
                for f in os.listdir(self.storage_dir) if f.endswith(".pt")]
