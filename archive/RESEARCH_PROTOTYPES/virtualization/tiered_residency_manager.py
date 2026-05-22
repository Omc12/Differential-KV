import time
from typing import Dict, Any

class TieredResidencyManager:
    """
    Manages layer residency across VRAM, Host RAM, and simulated SSD storage.
    Uses 'temperature' tracking to decide tiering.
    """
    def __init__(self, total_layers: int):
        self.total_layers = total_layers
        self.layer_temperatures = {i: 0.0 for i in range(total_layers)}
        self.last_access = {i: time.perf_counter() for i in range(total_layers)}

    def record_access(self, layer_idx: int):
        now = time.perf_counter()
        delta = now - self.last_access[layer_idx]
        # Increase temperature on access, decrease over time
        self.layer_temperatures[layer_idx] = (self.layer_temperatures[layer_idx] * 0.9) + 1.0
        self.last_access[layer_idx] = now

    def get_tier(self, layer_idx: int) -> str:
        temp = self.layer_temperatures[layer_idx]
        if temp > 5.0:
            return "VRAM"
        elif temp > 1.0:
            return "RAM"
        else:
            return "SSD" # [SIMULATED]

    def get_stats(self):
        tiers = [self.get_tier(i) for i in range(self.total_layers)]
        return {
            "vram_count": tiers.count("VRAM"),
            "ram_count": tiers.count("RAM"),
            "ssd_count": tiers.count("SSD")
        }
