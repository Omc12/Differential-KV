import hashlib
import os
import json
from typing import Dict, List, Optional, Any

class DeploymentIntegrityGuard:
    """
    Validates deployment coherence and ensures runtime reproducibility.
    Detects corruption in packaged artifacts.
    """
    def __init__(self, bundle_root: str):
        self.bundle_root = bundle_root

    def calculate_bundle_checksum(self, exclude_dirs: List[str] = None) -> str:
        """
        Generates a SHA256 checksum for the entire bundle.
        """
        sha256 = hashlib.sha256()
        exclude = exclude_dirs or ["dist", "session_checkpoints", "__pycache__", ".git"]
        
        for root, dirs, files in os.walk(self.bundle_root):
            dirs[:] = [d for d in dirs if d not in exclude]
            for names in sorted(files):
                filepath = os.path.join(root, names)
                with open(filepath, "rb") as f:
                    while True:
                        data = f.read(65536)
                        if not data:
                            break
                        sha256.update(data)
                        
        return sha256.hexdigest()

    def verify_reproducibility(self, expected_checksum: str) -> bool:
        """
        Verifies that the current deployment matches the expected state.
        """
        current = self.calculate_bundle_checksum()
        match = current == expected_checksum
        if not match:
            print(f"[DIG] Integrity mismatch! Expected: {expected_checksum}, Got: {current}")
        return match

    def validate_deployment_safety(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final safety check before production launch.
        """
        issues = []
        if config.get("runtime", {}).get("device") == "cuda":
            # Check if CUDA is actually available
            import torch
            if not torch.cuda.is_available():
                issues.append("CUDA requested but not available.")
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "integrity_score": 1.0 if len(issues) == 0 else 0.5
        }
