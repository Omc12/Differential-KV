import time
import uuid
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class SessionState:
    session_id: str
    created_at: float
    last_activity: float
    is_active: bool = True
    stream_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProductionSessionLifecycleManager:
    """
    OIS Phase 40.1: Production Session Lifecycle Manager.
    Manages session creation, cleanup, expiration, and tracking.
    """
    def __init__(self, expiry_seconds: int = 3600):
        self.sessions: Dict[str, SessionState] = {}
        self.expiry_seconds = expiry_seconds
        self.logger = logging.getLogger("SessionManager")

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            created_at=time.time(),
            last_activity=time.time(),
            metadata=metadata or {}
        )
        self.logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = time.time()
            return session
        return None

    def end_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            self.logger.info(f"Ended session: {session_id}")
            # In production, we might keep it for a while before full cleanup

    def cleanup_expired_sessions(self) -> int:
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() 
                   if now - s.last_activity > self.expiry_seconds]
        for sid in expired:
            del self.sessions[sid]
            self.logger.info(f"Cleaned up expired session: {sid}")
        return len(expired)

    def track_token(self, session_id: str, count: int = 1):
        session = self.get_session(session_id)
        if session:
            session.token_count += count

    def get_active_sessions_count(self) -> int:
        return sum(1 for s in self.sessions.values() if s.is_active)

    def handle_reconnect(self, session_id: str) -> bool:
        """Verifies if a session can be resumed."""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            self.logger.info(f"Session reconnected: {session_id}")
            session.last_activity = time.time()
            return True
        return False

    def list_orphan_sessions(self) -> List[str]:
        """Identifies sessions with no activity but marked active."""
        now = time.time()
        return [sid for sid, s in self.sessions.items() 
                if s.is_active and now - s.last_activity > 300] # 5 mins idle
