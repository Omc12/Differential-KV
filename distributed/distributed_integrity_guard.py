import torch
import hashlib
from typing import Dict, List, Any
import logging

class DistributedIntegrityGuard:
    """
    Distributed Symbolic Continuity and Deterministic Replay Validator.
    Ensures that KV segments remain consistent across migrations and remote access.
    """
    def __init__(self):
        self.segment_hashes: Dict[str, str] = {}
        self.continuity_violations: int = 0
        self.logger = logging.getLogger("DistributedIntegrityGuard")

    def compute_hash(self, tensor: torch.Tensor) -> str:
        """Computes a stable hash for a KV tensor."""
        # Ensure tensor is on CPU for hashing
        flat_data = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(flat_data).hexdigest()

    def register_integrity(self, segment_id: str, tensor: torch.Tensor):
        """Registers the expected state of a segment."""
        self.segment_hashes[segment_id] = self.compute_hash(tensor)
        self.logger.info(f"Registered integrity for {segment_id}")

    def validate_continuity(self, segment_id: str, tensor: torch.Tensor) -> bool:
        """Validates that a segment has not been corrupted during transfer."""
        if segment_id not in self.segment_hashes:
            self.logger.warning(f"No integrity record for {segment_id}")
            return True
        
        current_hash = self.compute_hash(tensor)
        if current_hash != self.segment_hashes[segment_id]:
            self.continuity_violations += 1
            self.logger.error(f"Symbolic continuity violation in {segment_id}!")
            return False
        
        return True

    def validate_deterministic_replay(self, segment_id: str, original_tensor: torch.Tensor, replayed_tensor: torch.Tensor) -> bool:
        """Ensures that distributed replay produces identical results."""
        match = torch.allclose(original_tensor, replayed_tensor, atol=1e-6)
        if not match:
            self.logger.error(f"Deterministic replay failed for {segment_id}!")
        return match

    def get_integrity_metrics(self) -> Dict[str, Any]:
        """Returns integrity and continuity metrics."""
        return {
            "remote_kv_integrity": 1.0 - (self.continuity_violations / max(1, len(self.segment_hashes))),
            "distributed_symbolic_continuity": len(self.segment_hashes) - self.continuity_violations,
            "continuity_violations": self.continuity_violations
        }
