import torch
from typing import Tuple

class SequentialAnchorLayout:
    """
    PHASE 7.5A: Sequential Anchor Layout
    Organizes anchor KV pairs in a contiguous memory block to maximize 
    GPU cache efficiency and minimize fragmentation during adaptive recovery.
    """
    def __init__(self, anchor_dim: int, device: str = "cuda"):
        self.anchor_dim = anchor_dim
        self.device = device
        self.max_anchors = 4096
        self.anchor_storage = torch.zeros((self.max_anchors, anchor_dim), device=device)
        self.current_count = 0

    def pack_anchors(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Packs adaptive anchors into a sequential buffer.
        """
        batch_size = keys.shape[0]
        if self.current_count + batch_size > self.max_anchors:
            # Simple circular buffer or expansion logic
            self.current_count = 0
            
        # Contiguous update
        self.anchor_storage[self.current_count:self.current_count + batch_size] = keys
        self.current_count += batch_size
        
        return self.anchor_storage[:self.current_count]

    def get_sequential_view(self) -> torch.Tensor:
        """
        Returns a contiguous view of all active anchors.
        """
        return self.anchor_storage[:self.current_count].contiguous()
