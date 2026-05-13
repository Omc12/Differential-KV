import torch
from runtime.adaptive_anchor_system.adaptive_anchor_modes import AdaptiveAnchorModes, AnchorSpacingMode
from runtime.adaptive_anchor_system.retrieval_density_mapper import RetrievalDensityMapper
from runtime.adaptive_anchor_system.context_entropy_mapper import ContextEntropyMapper
from runtime.adaptive_anchor_system.anchor_spacing_controller import AnchorSpacingController

class StructuredAnchorScheduler:
    def __init__(self):
        self.density_mapper = RetrievalDensityMapper()
        self.entropy_mapper = ContextEntropyMapper()
        self.spacing_controller = AnchorSpacingController()
        self.modes_engine = AdaptiveAnchorModes()
        
    def update_metrics(self, attn: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor, seq_len: int):
        self.density_mapper.update(indices, mask, seq_len)
        self.last_entropy_map = self.entropy_mapper.map_sequence_entropy(attn)

    def get_adaptive_anchors(self, seq_len: int) -> torch.Tensor:
        density_map = self.density_mapper.density_map
        layout = self.spacing_controller.get_structured_layout(
            seq_len, self.last_entropy_map if hasattr(self, 'last_entropy_map') else torch.tensor([]), density_map
        )
        device = density_map.device if density_map is not None else torch.device('cpu')
        all_anchors = []
        for start, end, mode in layout:
            chunk_anchors = self.modes_engine.calculate_anchor_indices(end - start, mode, device=device)
            all_anchors.append(chunk_anchors + start)
        return torch.cat(all_anchors).unique() if all_anchors else torch.tensor([], dtype=torch.long)

    def get_telemetry(self) -> dict:
        return {
            "avg_retrieval_density": self.density_mapper.density_map.mean().item() if self.density_mapper.density_map is not None else 1.0,
            "hotspot_count": len(self.density_mapper.identify_hotspots()),
            "current_layout": [mode.name for mode in self.spacing_controller.history]
        }
