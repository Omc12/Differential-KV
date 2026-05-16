import torch
import time
from typing import Dict, Any

class ServingResidencyTelemetry:
    """
    HSM System 3: Serving Residency Telemetry.
    Tracks material VRAM occupancy and stack-wide residency costs.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.history = []

    def sample_residency(self, active_sessions: int):
        """Samples current VRAM and residency state."""
        stats = {
            "timestamp": time.time(),
            "active_sessions": active_sessions
        }
        
        if self.device == "cuda":
            stats["vram_allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024**2)
            stats["vram_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024**2)
            stats["vram_max_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        else:
            stats["vram_allocated_mb"] = 0
            stats["vram_reserved_mb"] = 0
            
        self.history.append(stats)
        return stats

    def get_residency_report(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        
        avg_allocated = sum(h["vram_allocated_mb"] for h in self.history) / len(self.history)
        max_allocated = max(h["vram_allocated_mb"] for h in self.history)
        
        return {
            "avg_vram_allocated_mb": avg_allocated,
            "max_vram_allocated_mb": max_allocated,
            "peak_reserved_mb": max(h["vram_reserved_mb"] for h in self.history),
            "samples": len(self.history),
            "material_gpu_pressure": True if max_allocated > 1000 else False # Threshold for "heavy"
        }
