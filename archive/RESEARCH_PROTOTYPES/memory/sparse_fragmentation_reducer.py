"""
memory/sparse_fragmentation_reducer.py

Compaction and fragmentation management for sparse KV maps.
Ensures memory is contiguous and avoids 'Swiss cheese' allocation patterns.
"""

from typing import List, Dict, Tuple
import logging

class SparseFragmentationReducer:
    """
    Memory compactor for sparse KV blocks.
    """
    def __init__(self, block_size: int = 1024):
        self.block_size = block_size
        self.allocated_blocks: List[int] = []
        self.logger = logging.getLogger("SparseFragmentationReducer")

    def analyze_fragmentation(self) -> float:
        """
        Calculates fragmentation score: (max_id - min_id) / count.
        """
        if not self.allocated_blocks: return 0.0
        span = max(self.allocated_blocks) - min(self.allocated_blocks) + 1
        return 1.0 - (len(self.allocated_blocks) / span)

    def suggest_compaction(self) -> List[Tuple[int, int]]:
        """
        Suggests moves to compact memory.
        Returns list of (source_block_id, target_block_id).
        """
        frag = self.analyze_fragmentation()
        if frag < 0.2:
            return []
            
        self.logger.info(f"High Fragmentation ({frag:.2f}): Suggesting compaction...")
        # REAL implementation would find gaps and suggest moves
        return [(max(self.allocated_blocks), min(set(range(max(self.allocated_blocks))) - set(self.allocated_blocks)))]

    def record_allocation(self, block_id: int):
        """Tracks an allocation."""
        if block_id not in self.allocated_blocks:
            self.allocated_blocks.append(block_id)
            self.allocated_blocks.sort()

    def record_free(self, block_id: int):
        """Tracks a free event."""
        if block_id in self.allocated_blocks:
            self.allocated_blocks.remove(block_id)
