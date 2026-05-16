import torch
import os

class RealVRAMTracker:
    """
    Tracks true VRAM allocation and reserved memory.
    Solves the '0MB telemetry' issue by querying hardware directly.
    """
    def __init__(self, device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.history = []

    def get_current_vram(self) -> dict:
        """
        Returns real-time VRAM statistics in MB.
        """
        allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        
        stats = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
            "free_mb": (torch.cuda.get_device_properties(self.device).total_memory / (1024 * 1024)) - allocated
        }
        
        self.history.append(stats)
        return stats

    def track_tensor_vram(self, name: str, tensor: torch.Tensor) -> float:
        """Tracks the VRAM usage of a specific tensor."""
        size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
        return size_mb

    def reset_peak(self):
        """Resets the peak memory tracker."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def get_vram_drift(self) -> float:
        """Calculates VRAM fragmentation or drift over time."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1]["allocated_mb"] - self.history[0]["allocated_mb"]
