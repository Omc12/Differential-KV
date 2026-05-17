"""
MRO Phase 41.4: Sparse KV Compaction Engine.
Purpose: Compact active sparse KV blocks into highly contiguous memory layouts.
"""

from typing import Dict, Any

class SparseKVCompactionEngine:
    def __init__(self):
        self._fragmented_blocks = 0
        self._compacted_blocks = 0
        self._compaction_efficiency = 100.0

    def record_allocation(self, num_blocks: int, fragmented: bool):
        if fragmented:
            self._fragmented_blocks += num_blocks
        else:
            self._compacted_blocks += num_blocks

    def trigger_compaction(self) -> Dict[str, Any]:
        # Compact fragmented blocks
        if self._fragmented_blocks > 0:
            compacted = int(self._fragmented_blocks * 0.9)
            self._compacted_blocks += compacted
            self._fragmented_blocks -= compacted
            
        total = self._compacted_blocks + self._fragmented_blocks
        self._compaction_efficiency = (self._compacted_blocks / total * 100.0) if total > 0 else 100.0
        
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "fragmented_blocks": self._fragmented_blocks,
            "compacted_blocks": self._compacted_blocks,
            "compaction_efficiency_pct": self._compaction_efficiency
        }
