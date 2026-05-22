"""
memory/gpu_sparse_memory_mapper.py

Maps logical sparse indices to physical VRAM locations for optimized access.
Enforces memory alignment and residency policies.
"""

import torch
from typing import Dict, List

class GPUSparseMemoryMapper:
    def __init__(self, total_capacity: int, alignment: int = 128):
        self.capacity = total_capacity
        self.alignment = alignment
        self.free_slots = list(range(0, total_capacity, alignment))
        self.logical_to_physical = {}
        self.physical_to_logical = {}

    def map_sparse_kv(self, logical_id: int, size: int):
        """Maps a logical KV ID to a physical VRAM offset."""
        if not self.free_slots:
            return None
            
        physical_offset = self.free_slots.pop(0)
        self.logical_to_physical[logical_id] = physical_offset
        self.physical_to_logical[physical_offset] = logical_id
        return physical_offset

    def unmap_sparse_kv(self, logical_id: int):
        """Frees a physical VRAM offset associated with a logical ID."""
        if logical_id in self.logical_to_physical:
            physical_offset = self.logical_to_physical.pop(logical_id)
            del self.physical_to_logical[physical_offset]
            self.free_slots.append(physical_offset)
            self.free_slots.sort()
            return True
        return False

    def get_physical_layout(self):
        """Returns the current physical mapping for debugging and optimization."""
        return self.logical_to_physical.copy()

    def get_fragmentation_stats(self):
        """Returns fragmentation metrics for the mapped VRAM."""
        total_free = len(self.free_slots) * self.alignment
        return {
            "total_free_bytes": total_free,
            "utilization": 1.0 - (total_free / self.capacity),
            "slot_count": len(self.logical_to_physical)
        }
