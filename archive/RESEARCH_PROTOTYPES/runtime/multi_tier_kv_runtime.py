import torch
from typing import Dict, Tuple, Optional

class MultiTierKVRuntime:
    """
    Manages multi-tier KV cache storage (L1: VRAM, L2: System RAM).
    Optimizes for long-context throughput and hardware constraints.
    """
    def __init__(self, l1_capacity: int = 1024, l2_capacity: int = 4096):
        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        
        # KV storage: {layer_id: (keys, values)}
        self.l1_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.l2_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def update_kv(self, layer_id: int, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Update KV cache for a layer, managing transitions between tiers.
        """
        if layer_id not in self.l1_cache:
            self.l1_cache[layer_id] = (new_keys, new_values)
        else:
            old_keys, old_values = self.l1_cache[layer_id]
            updated_keys = torch.cat([old_keys, new_keys], dim=-2)
            updated_values = torch.cat([old_values, new_values], dim=-2)
            
            # Tiered overflow management
            if updated_keys.size(-2) > self.l1_capacity:
                overflow_len = updated_keys.size(-2) - self.l1_capacity
                l1_keys = updated_keys[:, :, -self.l1_capacity:, :]
                l1_vals = updated_values[:, :, -self.l1_capacity:, :]
                
                overflow_keys = updated_keys[:, :, :overflow_len, :]
                overflow_vals = updated_values[:, :, :overflow_len, :]
                
                self.l1_cache[layer_id] = (l1_keys, l1_vals)
                self._offload_to_l2(layer_id, overflow_keys, overflow_vals)
            else:
                self.l1_cache[layer_id] = (updated_keys, updated_values)

    def _offload_to_l2(self, layer_id: int, keys: torch.Tensor, values: torch.Tensor):
        """Move KV pairs to CPU RAM."""
        cpu_keys = keys.to("cpu")
        cpu_vals = values.to("cpu")
        
        if layer_id not in self.l2_cache:
            self.l2_cache[layer_id] = (cpu_keys, cpu_vals)
        else:
            old_k, old_v = self.l2_cache[layer_id]
            self.l2_cache[layer_id] = (
                torch.cat([old_k, cpu_keys], dim=-2),
                torch.cat([old_v, cpu_vals], dim=-2)
            )
            
        # Hard limit for L2
        if self.l2_cache[layer_id][0].size(-2) > self.l2_capacity:
            self.l2_cache[layer_id] = (
                self.l2_cache[layer_id][0][:, :, -self.l2_capacity:, :],
                self.l2_cache[layer_id][1][:, :, -self.l2_capacity:, :]
            )

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch active KV cache for computation."""
        return self.l1_cache.get(layer_id, (None, None))
