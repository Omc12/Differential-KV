import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

class InteractiveFailureReplaySystem:
    """
    ORX Phase 40.2: Interactive Failure Replay System.
    Captures and replays real operational failures for reproducibility.
    """
    def __init__(self, replay_dir: Path):
        self.replay_dir = replay_dir
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("FailureReplay")
        self.active_replays = []

    def capture_failure(self, failure_type: str, session_id: str, context: Dict[str, Any]):
        """Records a failure event for later replay."""
        failure_id = f"fail_{int(time.time())}_{session_id[:8]}"
        path = self.replay_dir / f"{failure_id}.json"
        
        record = {
            "failure_id": failure_id,
            "type": failure_type,
            "session_id": session_id,
            "context": context,
            "timestamp": time.time()
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        
        self.logger.info(f"Captured failure: {failure_id} ({failure_type})")

    def list_replays(self) -> List[str]:
        return [f.name for f in self.replay_dir.glob("*.json")]

    def replay_failure(self, failure_id: str) -> bool:
        """Simulates the re-occurrence of a recorded failure."""
        path = self.replay_dir / f"{failure_id}.json"
        if not path.exists():
            return False
            
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
            
        self.logger.warning(f"REPLAYING FAILURE: {record['type']} for session {record['session_id']}")
        # In a real system, this would trigger specific worker state overrides
        return True
