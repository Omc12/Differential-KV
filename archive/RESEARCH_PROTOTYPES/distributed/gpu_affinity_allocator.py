"""
distributed/gpu_affinity_allocator.py

Affinity-aware memory allocator for sparse KV blocks.
Ensures memory is allocated on the correct device for the shard owner.
"""

import torch
from typing import List, Dict, Optional, Tuple
import logging

class GPUAffinityAllocator:
    """
    Allocates and tracks GPU memory blocks with strict device affinity.
    """
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.allocations: Dict[int, Dict[int, torch.Tensor]] = {d: {} for d in device_ids}
        self.logger = logging.getLogger("GPUAffinityAllocator")

    def allocate_kv_block(
        self, 
        shard_id: int, 
        node_id: int, 
        shape: Tuple[int, ...], 
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Allocates a block for a specific shard on the designated node's device.
        """
        # In this implementation, node_id maps to device_id
        device_id = self.device_ids[node_id % len(self.device_ids)]
        
        self.logger.info(f"Allocating KV Block for Shard {shard_id} on Device {device_id}")
        
        with torch.cuda.device(device_id):
            block = torch.zeros(shape, dtype=dtype, device=f"cuda:{device_id}")
            self.allocations[device_id][shard_id] = block
            
        return block

    def get_block(self, shard_id: int, node_id: int) -> Optional[torch.Tensor]:
        """Retrieves an existing block with affinity check."""
        device_id = self.device_ids[node_id % len(self.device_ids)]
        return self.allocations[device_id].get(shard_id)

    def free_shard(self, shard_id: int, node_id: int):
        """Releases memory for a shard."""
        device_id = self.device_ids[node_id % len(self.device_ids)]
        if shard_id in self.allocations[device_id]:
            del self.allocations[device_id][shard_id]
            self.logger.info(f"Freed Shard {shard_id} from Device {device_id}")
