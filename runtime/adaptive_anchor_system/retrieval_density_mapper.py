import torch
from typing import Optional

class RetrievalDensityMapper:
    def __init__(self, window_size: int = 1024, decay: float = 0.99):
        self.window_size = window_size
        self.decay = decay
        self.density_map: Optional[torch.Tensor] = None

    def update(self, retrieval_indices: torch.Tensor, success_mask: torch.Tensor, total_seq_len: int):
        if self.density_map is None or self.density_map.size(0) < total_seq_len:
            new_map = torch.ones(total_seq_len, device=retrieval_indices.device)
            if self.density_map is not None:
                new_map[:self.density_map.size(0)] = self.density_map
            self.density_map = new_map

        self.density_map *= self.decay
        success_weights = success_mask.float()
        failure_weights = (1.0 - success_weights) * 2.0
        self.density_map.index_add_(0, retrieval_indices, -failure_weights)
        self.density_map.index_add_(0, retrieval_indices, success_weights * 0.1)
        self.density_map = torch.clamp(self.density_map, 0.0, 1.0)

    def identify_hotspots(self, threshold: float = 0.3) -> torch.Tensor:
        if self.density_map is None:
            return torch.tensor([], dtype=torch.long)
        return torch.where(self.density_map < threshold)[0]
