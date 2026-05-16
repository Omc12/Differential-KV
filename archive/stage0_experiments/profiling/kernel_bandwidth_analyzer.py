import torch
from typing import Dict

class KernelBandwidthAnalyzer:
    """
    PHASE 7.5B: Kernel Bandwidth Analyzer
    Calculates effective memory bandwidth utilization (GB/s) for sparse 
    kernels by comparing data moved (KV bytes) against kernel execution time.
    """
    def __init__(self):
        self.device_peak_gb_s = 1500.0 # e.g., H100/A100 class

    def measure_effective_bandwidth(
        self, 
        bytes_moved: int, 
        duration_ms: float
    ) -> Dict[str, float]:
        """
        Computes bandwidth metrics.
        """
        if duration_ms <= 0:
            return {"gb_s": 0.0, "efficiency": 0.0}
            
        # Bandwidth = Bytes / Time
        # GB/s = (Bytes / 10^9) / (Seconds)
        # GB/s = (Bytes / 10^6) / (ms)
        gb_s = (bytes_moved / 10**6) / duration_ms
        efficiency = (gb_s / self.device_peak_gb_s) * 100
        
        return {
            "effective_gb_s": gb_s,
            "peak_percent": efficiency,
            "total_data_mb": bytes_moved / 1024**2
        }

    def analyze_locality_impact(self, sequential_gb_s: float, sparse_gb_s: float) -> float:
        """Measures the 'sparse penalty' on bandwidth."""
        if sequential_gb_s == 0: return 0.0
        return 1.0 - (sparse_gb_s / sequential_gb_s)
