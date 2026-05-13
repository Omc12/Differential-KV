import torch

class AnchorCollisionReducer:
    def reduce_collisions(self, anchor_indices: torch.Tensor) -> torch.Tensor:
        if anchor_indices.numel() <= 1: return anchor_indices
        sorted_indices, _ = torch.sort(anchor_indices)
        diffs = sorted_indices[1:] - sorted_indices[:-1]
        keep_mask = torch.ones_like(sorted_indices, dtype=torch.bool)
        keep_mask[1:] = diffs >= 2
        return sorted_indices[keep_mask]
