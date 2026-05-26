import os
import torch
import json
import time
from typing import Dict, Any, Optional, List
import uuid

class ProductionSessionManager:
    """
    Handles multi-session lifecycle, persistence, and sparse residency management.
    Ensures sessions are correctly loaded, saved, and cleaned up.
    """
    def __init__(self, storage_path: str = "./session_checkpoints", max_resident_sessions: int = 5, kv_manager = None):
        self.storage_path = storage_path
        self.max_resident_sessions = max_resident_sessions
        self.kv_manager = kv_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.resident_sessions: List[str] = []  # LRU for VRAM residency
        self.message_histories: Dict[str, List[Dict[str, str]]] = {}  # session_id -> [{role, content}]

        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

    def create_session(self, config: Optional[Dict[str, Any]] = None) -> str:
        session_id = str(uuid.uuid4())
        session_metadata = {
            "session_id": session_id,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "config": config or {},
            "status": "active"
        }
        self.active_sessions[session_id] = session_metadata
        self._ensure_residency(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session["last_accessed"] = time.time()
            self._ensure_residency(session_id)
            return session
        
        # Try loading from disk
        return self._load_session(session_id)

    def save_session(self, session_id: str, sparse_state: Dict[str, torch.Tensor]):
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found.")
        
        path = os.path.join(self.storage_path, f"{session_id}.pt")
        torch.save(sparse_state, path)
        
        # Save metadata
        meta_path = os.path.join(self.storage_path, f"{session_id}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(self.active_sessions[session_id], f)

    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        meta_path = os.path.join(self.storage_path, f"{session_id}_meta.json")
        if not os.path.exists(meta_path):
            return None
            
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        self.active_sessions[session_id] = metadata
        self._ensure_residency(session_id)
        return metadata

    def _ensure_residency(self, session_id: str):
        if session_id in self.resident_sessions:
            # Move to end (most recently used)
            self.resident_sessions.remove(session_id)
            self.resident_sessions.append(session_id)
        else:
            if len(self.resident_sessions) >= self.max_resident_sessions:
                # Evict oldest
                evicted_id = self.resident_sessions.pop(0)
                self._evict_from_vram(evicted_id)
            
            self.resident_sessions.append(session_id)
            self._load_into_vram(session_id)

    def _evict_from_vram(self, session_id: str):
        if self.kv_manager is not None:
            try:
                # Take zero-copy snapshot under persisted ID
                self.kv_manager.snapshot_session(session_id, f"persisted_{session_id}")
                # Clear session blocks to free NativeBlockPool slots and GPU memory
                self.kv_manager.clear_session(session_id)
                print(f"[PSM] Evicted session {session_id} to VRAM snapshot.")
            except Exception as e:
                print(f"[PSM] Warning: failed to evict session {session_id} from VRAM: {e}")
        else:
            print(f"[PSM] Evicting session {session_id} from VRAM residency (no KV manager).")

    def _load_into_vram(self, session_id: str):
        if self.kv_manager is not None:
            checkpoint_id = f"persisted_{session_id}"
            if hasattr(self.kv_manager, "_session_checkpoints") and checkpoint_id in self.kv_manager._session_checkpoints:
                try:
                    self.kv_manager.restore_session(session_id, checkpoint_id)
                    self.kv_manager.delete_checkpoint(checkpoint_id)
                    print(f"[PSM] Loaded session {session_id} back from VRAM snapshot.")
                except Exception as e:
                    print(f"[PSM] Warning: failed to restore session {session_id} into VRAM: {e}")
        else:
            print(f"[PSM] Loading session {session_id} into VRAM residency (no KV manager).")

    def cleanup_idle_sessions(self, idle_timeout_seconds: int = 3600):
        current_time = time.time()
        to_delete = []
        for sid, meta in self.active_sessions.items():
            if current_time - meta["last_accessed"] > idle_timeout_seconds:
                to_delete.append(sid)
        
        for sid in to_delete:
            print(f"[PSM] Cleaning up idle session {sid}.")
            if sid in self.resident_sessions:
                self.resident_sessions.remove(sid)
            del self.active_sessions[sid]

    def list_sessions(self) -> List[str]:
        return list(self.active_sessions.keys())

    # ------------------------------------------------------------------
    # Conversation history management
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Returns accumulated message history for this session."""
        return self.message_histories.get(session_id, [])

    def append_message(self, session_id: str, role: str, content: str):
        """Appends a message to the session's conversation history."""
        if session_id not in self.message_histories:
            self.message_histories[session_id] = []
        self.message_histories[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str):
        """Clears the conversation history for a session (e.g. on explicit reset)."""
        self.message_histories.pop(session_id, None)
