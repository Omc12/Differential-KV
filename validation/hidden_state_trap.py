import torch
from typing import Any, Dict

class HiddenStateTrap:
    """
    Attempts to catch hidden state leakage by injecting 'marker' tensors
    and checking if they persist after a hard reset.
    """
    def __init__(self):
        self.trap_tensor = torch.tensor([1.23456789], dtype=torch.float64) # Unique value

    def inject_trap(self) -> torch.Tensor:
        """Returns the trap tensor to be placed in an execution context."""
        return self.trap_tensor

    def check_trap_presence(self, object_to_check: Any) -> bool:
        """Recursively checks if the trap tensor (or its value) is present in an object."""
        if isinstance(object_to_check, torch.Tensor):
            if torch.equal(object_to_check, self.trap_tensor):
                return True
        elif isinstance(object_to_check, dict):
            for v in object_to_check.values():
                if self.check_trap_presence(v): return True
        elif isinstance(object_to_check, list):
            for v in object_to_check:
                if self.check_trap_presence(v): return True
        return False

    def validate_reset(self, state_after_reset: Dict[str, Any]) -> bool:
        """Ensures the trap tensor did NOT survive the reset."""
        return not self.check_trap_presence(state_after_reset)
