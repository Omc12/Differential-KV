from typing import List, Tuple
from .hierarchical_memory_capsules import MemoryCapsule

class PathwayContinuityEngine:
    """
    PHASE 18.7C: Pathway Continuity Engine.
    Ensures that semantic bridges between high-fidelity capsules are preserved.
    Prevents 'islands of precision' from becoming disconnected.
    """
    def __init__(self, bridge_width: int = 16):
        self.bridge_width = bridge_width

    def identify_gaps(self, capsules: List[MemoryCapsule]) -> List[Tuple[int, int]]:
        """Identifies gaps between capsules that might need continuity bridges."""
        if len(capsules) < 2:
            return []
            
        sorted_capsules = sorted(capsules, key=lambda c: c.start_idx)
        gaps = []
        
        for i in range(len(sorted_capsules) - 1):
            curr = sorted_capsules[i]
            nxt = sorted_capsules[i+1]
            
            gap_size = nxt.start_idx - curr.end_idx
            if 0 < gap_size < 128: # Only bridge reasonably small gaps
                gaps.append((curr.end_idx, nxt.start_idx))
                
        return gaps

    def generate_bridge_indices(self, gap: Tuple[int, int]) -> List[int]:
        """Generates indices to preserve for a bridge across a gap."""
        start, end = gap
        mid = (start + end) // 2
        
        # Preserve a few tokens in the middle or sparsely across the gap
        bridge = list(range(start, min(start + self.bridge_width, end)))
        bridge.extend(range(max(start, end - self.bridge_width), end))
        return list(set(bridge))
