import torch
import time

class IOBandwidthMonitor:
    """
    Monitors and reports KV bandwidth and VRAM usage.
    Provides data for the 'bandwidth analysis' section of the final report.
    """
    def __init__(self):
        self.stats = []

    def log_step(self, kv_shape: torch.Size, sparsity: float, latency: float):
        """
        Logs metrics for a single inference step.
        """
        # Calculate theoretical VRAM in MB (assuming float16 = 2 bytes)
        elements = 1
        for dim in kv_shape:
            elements *= dim
        vram_mb = (elements * 2) / (1024 * 1024)
        
        # Bandwidth simulation: active data moved
        bandwidth_mb = vram_mb * (1.0 - sparsity)
        
        self.stats.append({
            "vram_mb": vram_mb,
            "sparsity": sparsity,
            "bandwidth_mb": bandwidth_mb,
            "latency_ms": latency * 1000
        })

    def get_summary(self):
        """Returns averaged stats."""
        if not self.stats:
            return {}
            
        avg_vram = sum(s["vram_mb"] for s in self.stats) / len(self.stats)
        avg_sparsity = sum(s["sparsity"] for s in self.stats) / len(self.stats)
        avg_bandwidth = sum(s["bandwidth_mb"] for s in self.stats) / len(self.stats)
        
        return {
            "avg_vram_mb": avg_vram,
            "avg_sparsity": avg_sparsity,
            "avg_bandwidth_mb": avg_bandwidth,
            "total_steps": len(self.stats)
        }
