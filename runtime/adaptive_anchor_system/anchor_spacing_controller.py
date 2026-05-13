import torch
from typing import List, Tuple
from runtime.adaptive_anchor_system.adaptive_anchor_modes import AnchorSpacingMode, AdaptiveAnchorModes

class AnchorSpacingController:
    def __init__(self, stability_window: int = 5):
        self.modes_engine = AdaptiveAnchorModes()
        self.history: List[AnchorSpacingMode] = []

    def determine_spacing(self, entropy: float, density: float) -> AnchorSpacingMode:
        mode = self.modes_engine.get_mode_for_metrics(density, entropy)
        self.history.append(mode)
        if len(self.history) > 5: self.history.pop(0)
        return min(self.history)

    def get_structured_layout(self, seq_len: int, entropy_map: torch.Tensor, density_map: torch.Tensor):
        chunk_size = 512
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        layout = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            e = entropy_map[i].item() if i < len(entropy_map) else 0.5
            d = density_map[start:end].mean().item() if density_map is not None else 1.0
            layout.append((start, end, self.determine_spacing(e, d)))
        return layout
