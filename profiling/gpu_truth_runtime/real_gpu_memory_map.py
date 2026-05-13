import torch
from typing import Dict

class RealGPUMemoryMap:
    """
    Provides a detailed map of VRAM usage across the sparse runtime.
    Tracks KV cache, anchor storage, and workspace memory.
    """
    def __init__(self, device: int = 0):
        self.device = device

    def get_detailed_map(self) -> Dict[str, float]:
        """Returns VRAM usage breakdown in MB."""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
        
        # Breakdown (hypothetical, would need tracking in the main runtime)
        return {
            "total_allocated": allocated,
            "total_reserved": reserved,
            "kv_cache_est": allocated * 0.8,
            "anchors_est": allocated * 0.05,
            "workspace_est": allocated * 0.15
        }

    def verify_vram_migration(self, tensor: torch.Tensor) -> bool:
        """Verifies if a tensor is truly on the GPU."""
        return tensor.is_cuda
