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
    def __init__(self,
                 storage_path: str = "./session_checkpoints",
                 max_resident_sessions: int = None,
                 kv_manager=None):
        # Default: 4 resident sessions to support concurrent workflows (like title generation) without swapping.
        # Override with DKV_MAX_SESSIONS env var for multi-user deployments.
        if max_resident_sessions is None:
            try:
                max_resident_sessions = int(os.environ.get("DKV_MAX_SESSIONS", "4"))
            except (ValueError, TypeError):
                max_resident_sessions = 4

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
            # Move to end (most recently used) only if not already at the end
            if self.resident_sessions[-1] != session_id:
                self.resident_sessions.remove(session_id)
                self.resident_sessions.append(session_id)
            # Already resident and up-to-date — nothing else needed
            return
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
            print(f"[PSM] Cleaning up idle session {sid} due to timeout.")
            self.delete_session(sid)

    def delete_session(self, session_id: str):
        """Permanently deletes the session and clears all associated history, VRAM blocks, checkpoints, and files."""
        print(f"[PSM] Deleting session {session_id} and freeing all VRAM/disk resources.")
        
        # 1. Clear conversation history
        self.clear_history(session_id)
        
        # 2. Remove from residency and active lists
        if session_id in self.resident_sessions:
            self.resident_sessions.remove(session_id)
        self.active_sessions.pop(session_id, None)
        
        # 3. Clear KV manager blocks
        if self.kv_manager is not None:
            try:
                self.kv_manager.clear_session(session_id)
                self.kv_manager.delete_checkpoint(f"persisted_{session_id}")
            except Exception as e:
                print(f"[PSM] Warning during deletion of session {session_id} from KV: {e}")
                
        # 4. Remove session checkpoint files from disk
        try:
            pt_path = os.path.join(self.storage_path, f"{session_id}.pt")
            if os.path.exists(pt_path):
                os.remove(pt_path)
            meta_path = os.path.join(self.storage_path, f"{session_id}_meta.json")
            if os.path.exists(meta_path):
                os.remove(meta_path)
        except Exception as e:
            print(f"[PSM] Warning during deletion of checkpoint files for session {session_id}: {e}")

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


class SharedPrefixManager:
    """
    Manages reference-counted shared prefix blocks to enable zero-overhead prefix sharing
    across multiple sessions (e.g., system prompts, few-shot templates).
    """
    def __init__(self, kv_manager=None):
        self.kv_manager = kv_manager
        # Maps prefix tuple of token IDs to dictionary containing:
        # - "pool_indices": list of allocated block pool indices
        # - "ref_count": integer reference count of sessions using it
        # - "anchor_indices": list of anchor indices for those blocks
        self.shared_prefixes = {}
        # Maps session_id -> list of prefix tuples it is currently using
        self.session_prefixes = {}

    def register_session_prefix(self, session_id: str, prefix_tokens: List[int], pool_indices: List[int], anchor_indices: List[int]):
        """
        Register a shared prefix for a session. Increments reference counts.
        """
        prefix_key = tuple(prefix_tokens)
        if prefix_key not in self.shared_prefixes:
            self.shared_prefixes[prefix_key] = {
                "pool_indices": pool_indices,
                "anchor_indices": anchor_indices,
                "ref_count": 0
            }
        
        self.shared_prefixes[prefix_key]["ref_count"] += 1
        
        if session_id not in self.session_prefixes:
            self.session_prefixes[session_id] = []
        self.session_prefixes[session_id].append(prefix_key)

    def release_session_prefixes(self, session_id: str):
        """
        Release all shared prefixes used by a session. Decrements reference counts,
        and frees pool blocks if reference count drops to 0.
        """
        if session_id not in self.session_prefixes:
            return
            
        prefixes = self.session_prefixes.pop(session_id)
        for prefix_key in prefixes:
            if prefix_key in self.shared_prefixes:
                self.shared_prefixes[prefix_key]["ref_count"] -= 1
                if self.shared_prefixes[prefix_key]["ref_count"] <= 0:
                    prefix_data = self.shared_prefixes.pop(prefix_key)
                    if self.kv_manager is not None and getattr(self.kv_manager, "native_pool", None) is not None:
                        for pool_idx in prefix_data["pool_indices"]:
                            try:
                                self.kv_manager.native_pool.free_block(pool_idx)
                            except Exception as e:
                                print(f"[SharedPrefixManager] Warning: failed to free shared block {pool_idx}: {e}")

    def lookup_prefix(self, prefix_tokens: List[int]):
        """
        Check if a prefix is already resident in the pool.
        """
        prefix_key = tuple(prefix_tokens)
        return self.shared_prefixes.get(prefix_key)
