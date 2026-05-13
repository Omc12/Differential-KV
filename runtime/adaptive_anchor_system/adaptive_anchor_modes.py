import enum
import torch
from dataclasses import dataclass

class AnchorSpacingMode(enum.IntEnum):
    ULTRA_DENSE = 64
    DENSE = 128
    BALANCED = 256
    SPARSE = 512

@dataclass
class AnchorConfig:
    mode: AnchorSpacingMode
    entropy_threshold: float
    retrieval_threshold: float
    warp_divergence_penalty: float = 0.1

class AdaptiveAnchorModes:
    """
    Manages the bounded anchor modes for structured adaptive spacing.
    Ensures GPU-friendly execution by sticking to power-of-two spacing buckets.
    """
    def __init__(self):
        self.modes = {
            AnchorSpacingMode.ULTRA_DENSE: AnchorConfig(AnchorSpacingMode.ULTRA_DENSE, 0.8, 0.90),
            AnchorSpacingMode.DENSE: AnchorConfig(AnchorSpacingMode.DENSE, 0.6, 0.95),
            AnchorSpacingMode.BALANCED: AnchorConfig(AnchorSpacingMode.BALANCED, 0.4, 0.98),
            AnchorSpacingMode.SPARSE: AnchorConfig(AnchorSpacingMode.SPARSE, 0.2, 0.99)
        }

    def get_mode_for_metrics(self, retrieval_density: float, context_entropy: float) -> AnchorSpacingMode:
        """
        Selects the optimal anchor spacing mode based on hardware truth metrics.
        """
        # High entropy or low retrieval density (meaning sparse collapse) requires denser anchors
        if context_entropy > 0.7 or retrieval_density < 0.85:
            return AnchorSpacingMode.ULTRA_DENSE
        elif context_entropy > 0.5 or retrieval_density < 0.92:
            return AnchorSpacingMode.DENSE
        elif context_entropy > 0.3 or retrieval_density < 0.97:
            return AnchorSpacingMode.BALANCED
        else:
            return AnchorSpacingMode.SPARSE

    def calculate_anchor_indices(self, seq_len: int, mode: AnchorSpacingMode, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Generates anchor indices with fixed spacing to prevent warp divergence.
        """
        spacing = int(mode)
        indices = torch.arange(0, seq_len, spacing, dtype=torch.long, device=device)
        return indices
