
import torch
from typing import Dict, Any, List, Optional

class PersistentHotpathManager:
    """
    PHASE 23.1: ELF - Persistent Hotpath Manager.
    Tracks stable activation corridors and symbolic hotpaths for reusable execution.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hotpath_cache = {}
        self.decay_rate = config.get("hotpath_decay", 0.95)
        
        self.metrics = {
            "hotpath_persistence_ratio": 0.0,
            "reusable_neighborhoods": 0,
            "hotpath_stability": 1.0
        }

    def update_hotpaths(self, active_indices: torch.Tensor, step: int):
        """
        Updates persistent hotpaths with currently active sparse indices.
        """
        # Convert indices to a stable 'corridor' representation
        path_id = f"step_{step % 10}" # Simplified for simulation
        
        if path_id not in self.hotpath_cache:
            self.hotpath_cache[path_id] = torch.zeros_like(active_indices, dtype=torch.float)
        
        # Ensure shape matches (seq_len grows during generation)
        if self.hotpath_cache[path_id].shape != active_indices.shape:
            # Simple resize/pad to match current active_indices
            new_cache = torch.zeros_like(active_indices, dtype=torch.float)
            # Copy old data where possible (truncated if smaller, padded if larger)
            min_L = min(self.hotpath_cache[path_id].shape[-1], active_indices.shape[-1])
            new_cache[..., :min_L] = self.hotpath_cache[path_id][..., :min_L]
            self.hotpath_cache[path_id] = new_cache
            
        # Accumulate activation frequency
        self.hotpath_cache[path_id] = self.hotpath_cache[path_id] * self.decay_rate + active_indices.float()
        
        # Calculate persistence: ratio of high-frequency indices that stay active
        high_freq = (self.hotpath_cache[path_id] > 0.5).float()
        persistence = torch.mean(high_freq).item()
        
        self.metrics["hotpath_persistence_ratio"] = 0.7 * self.metrics["hotpath_persistence_ratio"] + 0.3 * persistence
        self.metrics["reusable_neighborhoods"] = len(self.hotpath_cache)
        
        return high_freq > 0.5

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
