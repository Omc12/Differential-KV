import torch
from typing import List, Tuple, Optional
from .adaptive_anchor_modes import AdaptiveAnchorModes, AnchorSpacingMode
from .retrieval_density_mapper import RetrievalDensityMapper
from .context_entropy_mapper import ContextEntropyMapper
from .anchor_spacing_controller import AnchorSpacingController

class StructuredAnchorScheduler:
    """
    Main scheduler for Phase 7A - Structured Adaptive Anchor Recovery.
    Coordinates density mapping, entropy analysis, and structured spacing.
    """
    def __init__(self):
        self.density_mapper = RetrievalDensityMapper()
        self.entropy_mapper = ContextEntropyMapper()
        self.spacing_controller = AnchorSpacingController()
        self.modes_engine = AdaptiveAnchorModes()
        
    def update_metrics(self, 
                       attn_weights: torch.Tensor, 
                       retrieval_indices: torch.Tensor, 
                       success_mask: torch.Tensor,
                       total_seq_len: int):
        """
        Updates internal hardware truth maps.
        """
        self.density_mapper.update(retrieval_indices, success_mask, total_seq_len)
        # Entropy is updated per forward pass
        self.last_entropy_map = self.entropy_mapper.map_sequence_entropy(attn_weights)

    def get_adaptive_anchors(self, seq_len: int) -> torch.Tensor:
        """
        Calculates the complete set of adaptive anchors for the current context.
        """
        density_map = self.density_mapper.density_map
        layout = self.spacing_controller.get_structured_layout(
            seq_len, 
            self.last_entropy_map if hasattr(self, 'last_entropy_map') else torch.tensor([]),
            density_map
        )
        
        device = density_map.device if density_map is not None else torch.device('cpu')
        all_anchors = []
        for start, end, mode in layout:
            # Generate anchors for this region using the selected mode
            chunk_len = end - start
            chunk_anchors = self.modes_engine.calculate_anchor_indices(chunk_len, mode, device=device)
            all_anchors.append(chunk_anchors + start)
            
        if not all_anchors:
            return torch.tensor([], dtype=torch.long)
            
        return torch.cat(all_anchors).unique()

    def get_telemetry(self) -> dict:
        """Returns telemetry for Phase 7D validation."""
        return {
            "avg_retrieval_density": self.density_mapper.density_map.mean().item() if self.density_mapper.density_map is not None else 1.0,
            "hotspot_count": len(self.density_mapper.identify_hotspots()),
            "current_layout": [mode.name for mode in self.spacing_controller.history] if self.spacing_controller.history else []
        }
