import torch

class AnchorCollisionReducer:
    """
    Prevents anchor clusters from creating retrieval hotspots or sparse collisions.
    Ensures that adaptive density doesn't lead to redundant representation.
    """
    def __init__(self, collision_threshold: int = 2):
        self.collision_threshold = collision_threshold

    def reduce_collisions(self, anchor_indices: torch.Tensor, bucket_size: int = 8) -> torch.Tensor:
        """
        Filters anchors that are too close together in a way that causes sparse collisions.
        """
        if anchor_indices.numel() <= 1:
            return anchor_indices
            
        # Sort to handle proximity checks
        sorted_indices, _ = torch.sort(anchor_indices)
        
        # Calculate distances between adjacent anchors
        diffs = sorted_indices[1:] - sorted_indices[:-1]
        
        # If distance is too small (e.g., < 2), it might be a collision risk in sparse kernels
        # especially if we are in a high-density mode.
        # However, for structured spacing, we usually have fixed distances.
        # This is more relevant when merging hotspots with structured anchors.
        
        keep_mask = torch.ones_like(sorted_indices, dtype=torch.bool)
        # Simple heuristic: don't allow anchors within 2 tokens of each other 
        # unless they are critical sinks (handled elsewhere)
        keep_mask[1:] = diffs >= 2
        
        return sorted_indices[keep_mask]

    def detect_hotsets(self, retrieval_counts: torch.Tensor) -> torch.Tensor:
        """
        Identifies regions where too many queries are hitting the same anchors,
        leading to contention.
        """
        # [K] counts of how many times each token was retrieved
        hotsets = torch.where(retrieval_counts > self.collision_threshold)[0]
        return hotsets
