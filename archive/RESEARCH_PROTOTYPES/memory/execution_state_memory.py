import json
import os
from typing import Dict, Any, Optional

class ExecutionStateMemory:
    """
    Explicit serialized execution state memory.
    NO hidden activations are persisted. Only JSON-serializable engineering metrics and states.
    """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.state: Dict[str, Any] = {}
        
    def update_state(self, key: str, value: Any):
        """Updates the current execution state with a new key-value pair."""
        # Ensure value is JSON serializable to prevent hidden tensor persistence
        try:
            json.dumps(value)
            self.state[key] = value
        except (TypeError, OverflowError):
            raise ValueError(f"Value for {key} is not JSON serializable. Hidden state leakage blocked.")

    def save(self):
        """Serializes the state to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.state, f, indent=4)

    def load(self) -> bool:
        """Loads the state from disk."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                self.state = json.load(f)
            return True
        return False

    def clear(self):
        """Clears the state and deletes the storage file (Hard Reset)."""
        self.state = {}
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def get_summary(self) -> str:
        """Returns a string summary of the execution state for retrieval-linked summaries."""
        return json.dumps(self.state, sort_keys=True)
