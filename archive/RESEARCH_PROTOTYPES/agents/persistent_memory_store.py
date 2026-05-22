"""
agents/persistent_memory_store.py

Phase 12A: Persistent Multi-Session Memory
Provides a persistent storage layer for sparse memory anchors, allowing them
to survive across process restarts and multi-hour agent sessions.
"""

import os
import json
import torch
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path
from anchor_logic.semantic_anchor_system import SemanticAnchor, SemanticAnchorMemory

class PersistentMemoryStore:
    """
    Handles the physical persistence of SemanticAnchorMemory.
    Supports both metadata (JSON) and tensor (Pickle/Torch) storage.
    """
    def __init__(self, base_path: str = "session_checkpoints/memory"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_id: str, memory: SemanticAnchorMemory):
        """Saves the current state of memory for a given session."""
        session_path = self.base_path / session_id
        session_path.mkdir(exist_ok=True)

        # 1. Save metadata (everything except tensors)
        metadata = {
            "max_anchors": memory.max_anchors,
            "budget_per_token": memory.budget_per_token,
            "anchors": []
        }

        # 2. Save anchors
        for pos, anchor in memory.anchors.items():
            anchor_meta = {
                "token_id": anchor.token_id,
                "position": anchor.position,
                "importance_score": anchor.importance_score,
                "reason": anchor.reason,
                "metadata_only": anchor.metadata_only,
                "selected_heads": anchor.selected_heads,
                "metadata": anchor.metadata,
                "has_kv": anchor.kv_exact is not None
            }
            metadata["anchors"].append(anchor_meta)

            if anchor.kv_exact is not None:
                kv_path = session_path / f"anchor_{pos}.pt"
                torch.save(anchor.kv_exact, kv_path)

        with open(session_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[PersistentMemoryStore] Saved session '{session_id}' with {len(memory.anchors)} anchors.")

    def load_session(self, session_id: str) -> Optional[SemanticAnchorMemory]:
        """Loads memory for a given session."""
        session_path = self.base_path / session_id
        if not session_path.exists():
            print(f"[PersistentMemoryStore] Session '{session_id}' not found.")
            return None

        meta_path = session_path / "metadata.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        memory = SemanticAnchorMemory(
            max_anchors=metadata["max_anchors"],
            budget_per_token=metadata["budget_per_token"]
        )

        for anchor_meta in metadata["anchors"]:
            pos = anchor_meta["position"]
            kv_exact = None
            if anchor_meta["has_kv"]:
                kv_path = session_path / f"anchor_{pos}.pt"
                if kv_path.exists():
                    kv_exact = torch.load(kv_path, weights_only=True)

            anchor = SemanticAnchor(
                token_id=anchor_meta["token_id"],
                position=pos,
                kv_exact=kv_exact,
                importance_score=anchor_meta["importance_score"],
                reason=anchor_meta["reason"],
                metadata_only=anchor_meta["metadata_only"],
                selected_heads=anchor_meta["selected_heads"],
                metadata=anchor_meta["metadata"]
            )
            memory.add_anchor(anchor)

        print(f"[PersistentMemoryStore] Loaded session '{session_id}' with {len(memory.anchors)} anchors.")
        return memory

    def list_sessions(self) -> List[str]:
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def delete_session(self, session_id: str):
        session_path = self.base_path / session_id
        if session_path.exists():
            for f in session_path.glob("*"):
                f.unlink()
            session_path.rmdir()
