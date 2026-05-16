"""
kernels/gpu_resident_anchor_manager.py

Manager for GPU-resident anchors in Differential KV.
Ensures anchors remain in high-speed VRAM and manages their lifecycle.
"""

import torch

class GPUResidentAnchorManager:
    def __init__(self, d_model: int, max_anchors: int, device="cuda", dtype=torch.float16):
        self.d_model = d_model
        self.max_anchors = max_anchors
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate GPU memory for anchors
        self.anchors = torch.zeros((max_anchors, d_model), device=device, dtype=dtype)
        self.active_mask = torch.zeros(max_anchors, device=device, dtype=torch.bool)
        self.metadata = {} # To track anchor versions/importance

    def allocate_anchors(self, num_new_anchors: int):
        """Finds free slots and allocates them."""
        free_indices = torch.where(~self.active_mask)[0]
        if len(free_indices) < num_new_anchors:
            # Simple eviction logic: evict oldest (lowest indices) for now
            # In production, this would be more sophisticated
            self.active_mask[:num_new_anchors] = False
            free_indices = torch.where(~self.active_mask)[0]
            
        indices = free_indices[:num_new_anchors]
        self.active_mask[indices] = True
        return indices

    def update_anchors(self, indices: torch.Tensor, new_values: torch.Tensor):
        """Updates specific anchor values on-device."""
        self.anchors[indices] = new_values.to(self.dtype)

    def get_anchors(self, indices: torch.Tensor):
        """Retrieves anchors by indices."""
        return self.anchors[indices]

    def clear(self):
        """Resets all anchors."""
        self.anchors.zero_()
        self.active_mask.zero_()
        
    def get_residency_stats(self):
        """Returns statistics about anchor memory usage."""
        active_count = self.active_mask.sum().item()
        return {
            "active_anchors": active_count,
            "total_slots": self.max_anchors,
            "vram_usage_mb": (self.anchors.element_size() * self.anchors.numel()) / (1024 * 1024)
        }
