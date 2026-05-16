import torch

class KVBandwidthAnalyzer:
    """
    Analyzes KV movement bandwidth across PCIe and memory tiers.
    Identifies bottlenecks in hierarchical memory orchestration.
    """
    def __init__(self):
        self.transfers = []

    def record_transfer(self, size_bytes: int, duration_ms: float, source: str, target: str):
        bandwidth_gb_s = (size_bytes / 1024**3) / (duration_ms / 1000.0)
        self.transfers.append({
            "size_mb": size_bytes / 1024**2,
            "duration_ms": duration_ms,
            "bandwidth_gb_s": bandwidth_gb_s,
            "path": f"{source}->{target}"
        })

    def get_bottlenecks(self):
        """
        Identifies the slowest transfer paths.
        """
        return sorted(self.transfers, key=lambda x: x["bandwidth_gb_s"])
