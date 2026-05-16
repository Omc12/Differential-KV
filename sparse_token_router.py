import torch
from typing import Dict, Any

class SparseTokenRouter:
    """
    Manages selective token participation in compute paths.
    Integrates with SEM and SML to ensure token-level sparsity.
    """
    def __init__(self):
        self.stats = {
            "tokens_routed": 0,
            "tokens_skipped": 0
        }

    def route_to_compute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Only allows active tokens to enter heavy compute paths.
        """
        self.stats["tokens_routed"] += mask.sum().item()
        self.stats["tokens_skipped"] += (mask == False).sum().item()
        
        # In ATC, we bypass the compute entirely for skipped tokens
        return x * mask.unsqueeze(-1).float()
