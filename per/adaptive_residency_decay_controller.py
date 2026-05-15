
import torch
from typing import Dict, Any, List, Optional

class AdaptiveResidencyDecayController:
    """
    PHASE 23.2: PER - Adaptive Residency Decay Controller.
    Regulates residency lifetime and cooling schedules to prevent bloat.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_temperature = 1.0
        
        self.metrics = {
            "residency_decay_health": 1.0,
            "stale_region_eviction_rate": 0.0,
            "cooling_efficiency": 1.0
        }

    def regulate_decay(self, current_vram_usage: int, max_vram: int):
        """
        Adjusts decay rates based on system pressure.
        """
        pressure = current_vram_usage / (max_vram + 1e-9)
        
        if pressure > 0.8:
            # High pressure: speed up decay (heat up)
            self.global_temperature = min(2.0, self.global_temperature + 0.1)
        else:
            # Low pressure: slow down decay (cool down)
            self.global_temperature = max(0.5, self.global_temperature - 0.05)
            
        self.metrics["residency_decay_health"] = 1.0 - (pressure * 0.2)
        self.metrics["cooling_efficiency"] = 1.0 / (self.global_temperature + 1e-9)
        
        return self.global_temperature

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
