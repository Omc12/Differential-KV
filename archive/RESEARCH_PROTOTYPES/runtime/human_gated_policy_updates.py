from runtime.runtime_policy_registry import RuntimePolicyRegistry
from typing import Dict, Any, Optional, List

class HumanGatedPolicyUpdates:
    """
    Approval gate logic for policy changes.
    Ensures critical policy shifts require explicit approval.
    """
    def __init__(self, registry: RuntimePolicyRegistry):
        self.registry = registry
        self.pending_updates = []

    def propose_update(self, name: str, new_value: Any, reason: str):
        """Proposes an update that must be approved."""
        self.pending_updates.append({
            "name": name,
            "new_value": new_value,
            "reason": reason
        })

    def approve_update(self, index: int = 0) -> bool:
        """Approves a pending update."""
        if 0 <= index < len(self.pending_updates):
            update = self.pending_updates.pop(index)
            self.registry.update_policy(update["name"], update["new_value"], f"APPROVED: {update['reason']}")
            return True
        return False

    def reject_update(self, index: int = 0):
        """Rejects a pending update."""
        if 0 <= index < len(self.pending_updates):
            self.pending_updates.pop(index)

    def get_pending(self) -> List[Dict[str, Any]]:
        return self.pending_updates
