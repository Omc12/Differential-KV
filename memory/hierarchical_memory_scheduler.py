import torch
from typing import Dict, List

class HierarchicalMemoryScheduler:
    """
    Manages KV distribution across memory tiers (VRAM, RAM, SSD).
    Optimizes for retrieval latency and context capacity.
    """
    def __init__(self, vram_limit_gb: float = 12.0):
        self.vram_limit_bytes = vram_limit_gb * 1024**3
        self.tiers = {
            "vram": {"capacity": self.vram_limit_bytes, "used": 0, "latency": 0.001},
            "ram": {"capacity": 64 * 1024**3, "used": 0, "latency": 0.1},
            "ssd": {"capacity": 1024 * 1024**3, "used": 0, "latency": 10.0}
        }

    def allocate(self, size_bytes: int, priority: float) -> str:
        """
        Decides which tier to allocate memory in based on priority.
        Priority > 0.8 goes to VRAM, else RAM.
        """
        if priority > 0.8 and self.tiers["vram"]["used"] + size_bytes < self.tiers["vram"]["capacity"]:
            self.tiers["vram"]["used"] += size_bytes
            return "vram"
        elif self.tiers["ram"]["used"] + size_bytes < self.tiers["ram"]["capacity"]:
            self.tiers["ram"]["used"] += size_bytes
            return "ram"
        else:
            self.tiers["ssd"]["used"] += size_bytes
            return "ssd"

    def release(self, size_bytes: int, tier: str):
        self.tiers[tier]["used"] = max(0, self.tiers[tier]["used"] - size_bytes)
