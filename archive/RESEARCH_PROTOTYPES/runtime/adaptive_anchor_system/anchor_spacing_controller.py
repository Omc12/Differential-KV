import torch
from typing import List, Tuple
from .adaptive_anchor_modes import AnchorSpacingMode, AdaptiveAnchorModes

class AnchorSpacingController:
    """
    Coordinates between entropy and retrieval density to select optimal anchor spacing.
    Prevents frequent oscillations between modes.
    """
    def __init__(self, stability_window: int = 5):
        self.modes_engine = AdaptiveAnchorModes()
        self.history: List[AnchorSpacingMode] = []
        self.stability_window = stability_window

    def determine_spacing(self, 
                          region_entropy: float, 
                          region_retrieval_density: float) -> AnchorSpacingMode:
        """
        Calculates the required spacing for a region.
        """
        target_mode = self.modes_engine.get_mode_for_metrics(region_retrieval_density, region_entropy)
        
        # Apply stability logic
        self.history.append(target_mode)
        if len(self.history) > self.stability_window:
            self.history.pop(0)
            
        # Use majority vote or 'most conservative' (densest) in window
        # To prevent retrieval collapse, we favor denser anchors if they were recently needed
        densest_mode = min(self.history) # IntEnum: smaller values are denser (64 < 128)
        
        return densest_mode

    def get_structured_layout(self, 
                               seq_len: int, 
                               entropy_map: torch.Tensor, 
                               density_map: torch.Tensor) -> List[Tuple[int, int, AnchorSpacingMode]]:
        """
        Divides the sequence into chunks and assigns a spacing mode to each.
        """
        # Chunk size matches the largest spacing bucket or a fixed granularity
        chunk_size = 512 
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        layout = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            
            # Map indices to maps (which might be lower res)
            # For simplicity, assume maps are token-level or we sample them
            chunk_entropy = entropy_map[i].item() if i < len(entropy_map) else 0.5
            chunk_density = density_map[start:end].mean().item() if density_map is not None else 1.0
            
            mode = self.determine_spacing(chunk_entropy, chunk_density)
            layout.append((start, end, mode))
            
        return layout
