import torch
import hashlib
from typing import Dict, Any, List
import logging

class CommunicationIntegrityGuard:
    """
    Ensures deterministic replay and synchronization-safe cognition migration.
    """
    def __init__(self):
        self.transfer_log: List[Dict] = []
        self.corruption_events = 0
        self.logger = logging.getLogger("CommunicationIntegrityGuard")

    def validate_transfer(self, segment_id: str, original_hash: str, received_tensor: torch.Tensor) -> bool:
        """Validates that a transferred tensor matches its original hash."""
        # Ensure tensor is on CPU for hashing
        flat_data = received_tensor.detach().cpu().numpy().tobytes()
        received_hash = hashlib.sha256(flat_data).hexdigest()
        
        if original_hash != received_hash:
            self.corruption_events += 1
            self.logger.error(f"Transfer corruption in {segment_id}!")
            return False
        
        self.transfer_log.append({
            "segment_id": segment_id,
            "hash": received_hash,
            "status": "success"
        })
        return True

    def get_integrity_metrics(self) -> Dict[str, Any]:
        return {
            "distributed_symbolic_continuity": 1.0 - (self.corruption_events / max(1, len(self.transfer_log))),
            "transfer_corruption_count": self.corruption_events,
            "total_validated_transfers": len(self.transfer_log)
        }

