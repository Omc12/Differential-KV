from typing import List, Dict, Any

class DensePathResidencyAuditor:
    """
    Verifies that all required dense paths are active during production decode.
    Prevents fake production execution.
    """
    def __init__(self):
        self.active_paths = set()

    def audit_step(self, path: str):
        self.active_paths.add(path)

    def verify_full_path(self) -> bool:
        required = {"embeddings", "mlp", "logits", "sampling", "tokenizer"}
        missing = required - self.active_paths
        if missing:
            print(f"[FRM] AUDIT FAILURE: Missing dense paths: {missing}")
            return False
        return True

    def get_audit_report(self) -> Dict[str, Any]:
        return {
            "active_dense_paths": sorted(list(self.active_paths)),
            "full_path_materialized": self.verify_full_path()
        }

# Global singleton
auditor = DensePathResidencyAuditor()
