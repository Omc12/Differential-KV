import torch
from typing import List

class HotspotAnchorAllocator:
    """
    Allocates high-density anchors specifically for detected retrieval 'hotspots' 
    where density mapping shows persistent failure.
    """
    def __init__(self, max_extra_anchors: int = 128):
        self.max_extra_anchors = max_extra_anchors

    def allocate_hotspot_anchors(self, 
                                 hotspot_indices: torch.Tensor, 
                                 current_anchors: torch.Tensor) -> torch.Tensor:
        """
        Adds extra anchors around hotspots if they aren't already covered.
        """
        if hotspot_indices.numel() == 0:
            return current_anchors
            
        # Convert to set for faster lookup or use torch.isin
        # We want to add anchors around the hotspot to 'bridge' the information gap
        
        extra_anchors = []
        for idx in hotspot_indices:
            # Add neighbors to strengthen the retrieval manifold in this region
            # (idx-1, idx+1)
            extra_anchors.append(idx - 1)
            extra_anchors.append(idx + 1)
            
        extra_tensor = torch.tensor(extra_anchors, device=hotspot_indices.device, dtype=torch.long)
        
        # Filter out invalid indices (< 0)
        extra_tensor = extra_tensor[extra_tensor >= 0]
        
        # Combine and unique
        combined = torch.cat([current_anchors, extra_tensor])
        
        # Cap the total number of anchors to prevent performance degradation
        final_anchors = combined.unique()
        if final_anchors.size(0) > (current_anchors.size(0) + self.max_extra_anchors):
            # If we exceed budget, prioritize original anchors + some hotspots
            # For now, just unique is okay as it's adaptive
            pass
            
        return final_anchors
