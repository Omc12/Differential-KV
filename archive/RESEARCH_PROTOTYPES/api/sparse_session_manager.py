import uuid
import time
from typing import Dict, Any, Optional

class SparseSessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, config: Optional[Dict[str, Any]] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_used": time.time(),
            "config": config or {},
            "kv_cache_ref": None, # In a real implementation, this would point to the GPU handle
            "history": []
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id in self.sessions:
            self.sessions[session_id]["last_used"] = time.time()
            return self.sessions[session_id]
        return None

    def update_session(self, session_id: str, new_history_item: Dict[str, str]):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append(new_history_item)
            self.sessions[session_id]["last_used"] = time.time()

    def list_sessions(self):
        return list(self.sessions.keys())

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
