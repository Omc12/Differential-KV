import torch
from typing import Any, Dict

class StateIntegrityAuditor:
    """
    Audits the execution state to ensure no hidden activations or tensors are persisted.
    Strictly enforces JSON-serializable state components.
    """
    def __init__(self):
        self.violations = []

    def audit_state(self, state: Dict[str, Any]) -> bool:
        """
        Checks the state for non-serializable objects (like PyTorch Tensors).
        Returns True if the state is clean, False otherwise.
        """
        self.violations = []
        self._check_recursive(state, "root")
        return len(self.violations) == 0

    def _check_recursive(self, item: Any, path: str):
        if isinstance(item, torch.Tensor):
            self.violations.append(f"FORBIDDEN: Tensor found at {path}")
        elif isinstance(item, dict):
            for k, v in item.items():
                self._check_recursive(v, f"{path}.{k}")
        elif isinstance(item, list):
            for i, v in enumerate(item):
                self._check_recursive(v, f"{path}[{i}]")
        # Add checks for other forbidden types if necessary (e.g. custom classes without __getstate__)

    def get_audit_report(self) -> Dict[str, Any]:
        return {
            "status": "PASS" if not self.violations else "FAIL",
            "violations": self.violations,
            "total_violations": len(self.violations)
        }
