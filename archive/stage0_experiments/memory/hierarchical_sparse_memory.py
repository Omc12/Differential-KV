"""
memory/hierarchical_sparse_memory.py

Phase 12C: Hierarchical Sparse Memory
Extends the Semantic Anchor Memory to support million-token contexts by
organizing anchors into hierarchical tiers (L1, L2, L3).
"""

from typing import Dict, List, Optional
import torch
from anchor_logic.semantic_anchor_system import SemanticAnchor, SemanticAnchorMemory

class HierarchicalSparseMemory:
    """
    Manages three tiers of memory:
    - L1 (VRAM): Active anchors, lowest latency.
    - L2 (RAM): Cached anchors, medium latency.
    - L3 (Disk): Persistent anchors, highest latency.
    """
    def __init__(self, l1_max=256, l2_max=2048, l3_max=16384):
        self.l1_memory = SemanticAnchorMemory(max_anchors=l1_max)
        self.l2_memory: Dict[int, SemanticAnchor] = {}
        self.l3_memory: Dict[int, SemanticAnchor] = {}
        self.l2_max = l2_max
        self.l3_max = l3_max

    def add_anchor(self, anchor: SemanticAnchor):
        """Adds a new anchor, potentially triggering tier migrations."""
        if len(self.l1_memory.anchors) < self.l1_memory.max_anchors:
            self.l1_memory.add_anchor(anchor)
        else:
            # Promote current L1 tail to L2
            # (Simplified: just add to L2)
            self._move_to_l2(anchor)

    def _move_to_l2(self, anchor: SemanticAnchor):
        if len(self.l2_memory) >= self.l2_max:
            # Evict from L2 to L3
            evict_pos = min(self.l2_memory.keys(), key=lambda k: self.l2_memory[k].importance_score)
            self._move_to_l3(self.l2_memory.pop(evict_pos))
        
        # Ensure tensor is in RAM, not VRAM
        if anchor.kv_exact is not None:
             anchor.kv_exact = anchor.kv_exact.cpu()
        self.l2_memory[anchor.position] = anchor

    def _move_to_l3(self, anchor: SemanticAnchor):
        if len(self.l3_memory) >= self.l3_max:
            # Permanent eviction
            evict_pos = min(self.l3_memory.keys(), key=lambda k: self.l3_memory[k].importance_score)
            del self.l3_memory[evict_pos]
        
        # In a real system, we'd serialize to disk here
        self.l3_memory[anchor.position] = anchor

    def get_anchor(self, position: int) -> Optional[SemanticAnchor]:
        """Retrieves an anchor from any tier, promoting it to L1 if found in L2/L3."""
        if position in self.l1_memory.anchors:
            return self.l1_memory.anchors[position]
        
        if position in self.l2_memory:
            anchor = self.l2_memory.pop(position)
            # Move KV back to GPU
            if anchor.kv_exact is not None:
                anchor.kv_exact = anchor.kv_exact.cuda() if torch.cuda.is_available() else anchor.kv_exact
            self.l1_memory.add_anchor(anchor)
            return anchor
            
        if position in self.l3_memory:
            anchor = self.l3_memory.pop(position)
            self.l1_memory.add_anchor(anchor)
            return anchor
            
        return None
