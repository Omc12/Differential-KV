import torch

class KernelBandwidthAnalyzer:
    """
    Analyzes the effective bandwidth utilization of sparse kernels.
    Compares real-world throughput against theoretical peak.
    """
    def __init__(self):
        self.peak_bandwidth = 900 # GB/s (example for A100)

    def calculate_utilization(self, bytes_transferred: int, duration_ms: float) -> float:
        """
        Calculates % of peak bandwidth utilization.
        """
        if duration_ms <= 0:
            return 0.0
            
        gbps = (bytes_transferred / (1024**3)) / (duration_ms / 1000.0)
        utilization = (gbps / self.peak_bandwidth) * 100
        return utilization

    def analyze_access_pattern(self, indices: torch.Tensor) -> str:
        """Heuristic for access pattern quality."""
        if indices.numel() < 2:
            return "SEQUENTIAL"
        
        diffs = indices[1:] - indices[:-1]
        if torch.all(diffs == 1):
            return "SEQUENTIAL"
        elif torch.all(diffs < 32):
            return "COALESCED"
        else:
            return "RANDOM"
