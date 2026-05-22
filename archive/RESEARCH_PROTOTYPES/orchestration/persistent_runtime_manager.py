import json
import os
import time
import logging

logger = logging.getLogger(__name__)

class PersistentRuntimeManager:
    """
    Manages long-horizon persistence for cognitive runtimes. Handles checkpointing,
    session resumption, and preventing entropy deaths over multi-day sessions.
    """
    def __init__(self, checkpoint_dir: str = "./session_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.active_sessions = {}

    def start_persistent_session(self, session_id: str) -> bool:
        """Starts a new persistent cognitive session."""
        if session_id in self.active_sessions:
            logger.warning(f"Session {session_id} is already active.")
            return False
            
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "last_checkpoint": time.time(),
            "status": "running",
            "entropy_level": 0.1
        }
        logger.info(f"Started persistent session {session_id}")
        return True

    def checkpoint_session(self, session_id: str, latent_state: dict):
        """Saves the latent manifold state to disk for crash recovery."""
        if session_id not in self.active_sessions:
            return
            
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{session_id}.ckpt")
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "state": latent_state,
                    "entropy": self.active_sessions[session_id]["entropy_level"]
                }, f)
            self.active_sessions[session_id]["last_checkpoint"] = time.time()
            logger.info(f"Checkpoint saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to checkpoint session {session_id}: {e}")

    def resume_session(self, session_id: str) -> dict:
        """Attempts to restore a session from the latest checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{session_id}.ckpt")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found for {session_id}")
            return None
            
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            self.active_sessions[session_id] = {
                "start_time": data["timestamp"],
                "last_checkpoint": data["timestamp"],
                "status": "resumed",
                "entropy_level": data["entropy"]
            }
            logger.info(f"Resumed session {session_id} from checkpoint.")
            return data["state"]
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return None

if __name__ == "__main__":
    manager = PersistentRuntimeManager()
    manager.start_persistent_session("agent-alpha-7d")
    manager.checkpoint_session("agent-alpha-7d", {"manifold_id": "m1", "vectors": [0.1, 0.2]})
