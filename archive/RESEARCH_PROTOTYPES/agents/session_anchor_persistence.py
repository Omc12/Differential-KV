"""
agents/session_anchor_persistence.py

Phase 12A: Session Anchor Persistence
Orchestrates the lifecycle of anchors within a specific user session, 
handling periodic auto-saves and state synchronization.
"""

import time
import threading
from typing import Optional
from agents.persistent_memory_store import PersistentMemoryStore
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory

class SessionAnchorPersistence:
    """
    Manages session-level persistence for an active agent.
    Provides background auto-saving and manual state capture.
    """
    def __init__(self, session_id: str, memory: SemanticAnchorMemory, auto_save_interval: int = 300):
        self.session_id = session_id
        self.memory = memory
        self.store = PersistentMemoryStore()
        self.auto_save_interval = auto_save_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_auto_save(self):
        """Starts a background thread for periodic persistence."""
        if self._thread is not None:
            return

        def _run():
            while not self._stop_event.is_set():
                time.sleep(self.auto_save_interval)
                if not self._stop_event.is_set():
                    self.checkpoint()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print(f"[SessionAnchorPersistence] Started auto-save for session '{self.session_id}' (interval: {self.auto_save_interval}s)")

    def stop_auto_save(self):
        """Stops the auto-save thread."""
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            print(f"[SessionAnchorPersistence] Stopped auto-save for session '{self.session_id}'")

    def checkpoint(self):
        """Manually triggers a save of the current memory state."""
        start_time = time.time()
        self.store.save_session(self.session_id, self.memory)
        elapsed = time.time() - start_time
        print(f"[SessionAnchorPersistence] Checkpoint completed for '{self.session_id}' in {elapsed:.2f}s")

    def restore(self) -> bool:
        """Restores memory from the persistent store."""
        loaded_memory = self.store.load_session(self.session_id)
        if loaded_memory:
            # Transfer anchors to the current memory instance
            self.memory.reset()
            for pos, anchor in loaded_memory.anchors.items():
                self.memory.add_anchor(anchor)
            return True
        return False
