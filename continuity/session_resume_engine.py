import torch
from typing import Dict, Any, Optional
from .cross_session_bridge import CrossSessionBridge
from .persistent_goal_tracker import PersistentGoalTracker
from identity.persistent_cognitive_identity import PersistentCognitiveIdentity

class SessionResumeEngine:
    """
    Orchestrates the restoration of cognitive states and objectives.
    Ensures seamless transition between execution sessions.
    """
    def __init__(self, identity_manager: PersistentCognitiveIdentity):
        self.bridge = CrossSessionBridge()
        self.goal_tracker = PersistentGoalTracker()
        self.identity_manager = identity_manager

    def resume_last_session(self) -> Optional[Dict[str, Any]]:
        """
        Attempts to resume the most recent session.
        """
        sessions = self.bridge.list_checkpoints()
        if not sessions:
            print("No previous sessions found.")
            return None
            
        # Assuming sessions are named in a way that we can find the latest
        latest_session = sorted(sessions)[-1]
        return self.resume_session(latest_session)

    def resume_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Resumes a specific session and restores identity and goals.
        """
        state = self.bridge.restore_session(session_id)
        if state is None:
            return None
            
        # Restore identity if present in state
        if "identity_id" in state:
            try:
                self.identity_manager.load_identity(state["identity_id"])
            except Exception as e:
                print(f"Failed to restore identity: {e}")
                
        # Get active goals to re-prioritize
        active_goals = self.goal_tracker.get_active_goals()
        state["active_goals"] = active_goals
        
        return state

    def prepare_shutdown(self, session_id: str, current_state: Dict[str, Any]):
        """
        Prepares for session shutdown by saving state and identity.
        """
        # Ensure identity is saved
        self.identity_manager.save_identity()
        
        # Add identity_id to state
        if self.identity_manager.current_identity_id:
            current_state["identity_id"] = self.identity_manager.current_identity_id
            
        # Checkpoint session
        self.bridge.checkpoint_session(session_id, current_state)
