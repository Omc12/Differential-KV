"""
agents/cross_session_anchor_sync.py

Handles persistent anchor synchronization across agent sessions.
Ensures that code-level 'knowledge' persists beyond a single process lifetime.
"""

import torch
import os
import json
from typing import Dict, Any, List, Optional
import logging

class CrossSessionAnchorSync:
    """
    Serializes and restores sparse KV anchors between agent sessions.
    """
    def __init__(self, persistence_dir: str):
        self.persistence_dir = persistence_dir
        os.makedirs(persistence_dir, exist_ok=True)
        self.logger = logging.getLogger("CrossSessionAnchorSync")

    def save_session_anchors(self, session_id: str, anchors: torch.Tensor, metadata: List[Dict[str, Any]]):
        """
        Saves session anchors to disk for future restoration.
        """
        path = os.path.join(self.persistence_dir, f"{session_id}_anchors.pt")
        meta_path = os.path.join(self.persistence_dir, f"{session_id}_metadata.json")
        
        torch.save(anchors, path)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Session {session_id} anchors persisted to {path}")

    def load_session_anchors(self, session_id: str) -> Optional[tuple]:
        """
        Restores anchors from a previous session.
        """
        path = os.path.join(self.persistence_dir, f"{session_id}_anchors.pt")
        meta_path = os.path.join(self.persistence_dir, f"{session_id}_metadata.json")
        
        if not os.path.exists(path):
            return None
            
        anchors = torch.load(path)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        return anchors, metadata

    def get_available_sessions(self) -> List[str]:
        """Returns a list of session IDs available for restoration."""
        return [f.replace("_anchors.pt", "") for f in os.listdir(self.persistence_dir) if f.endswith("_anchors.pt")]
