from enum import Enum
from typing import List, Dict, Tuple
import uuid

class CapsuleScale(Enum):
    MICRO = 1  # Exact symbolic (IDs, fragments)
    MESO = 2   # Local semantic context
    MACRO = 3  # Global instruction / continuity

class MemoryCapsule:
    def __init__(self, start_idx, end_idx, scale: CapsuleScale, relevance_score: float):
        self.capsule_id = f"hmc_{scale.name.lower()}_{uuid.uuid4().hex[:6]}"
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.scale = scale
        self.relevance_score = relevance_score
        self.precision_tier = self._determine_tier()

    def _determine_tier(self):
        if self.scale == CapsuleScale.MICRO:
            return "HIGH"
        elif self.scale == CapsuleScale.MESO:
            return "MEDIUM"
        return "LOW"

class MultiScaleCapsuleHierarchy:
    """
    PHASE 18.8C: Multi-Scale Memory Capsules.
    Manages Micro, Meso, and Macro capsules to preserve symbolic and semantic structure.
    """
    def __init__(self):
        self.active_capsules: Dict[str, MemoryCapsule] = {}

    def allocate_capsules(self, chunk_spans, scale: CapsuleScale):
        """
        Allocates capsules for detected spans at a specific scale.
        """
        new_capsules = []
        for start, end, score in chunk_spans:
            cap = MemoryCapsule(start, end, scale, score)
            self.active_capsules[cap.capsule_id] = cap
            new_capsules.append(cap)
        return new_capsules

    def get_token_mask(self, seq_len):
        """
        Returns a mask indicating which tokens are protected by which scale.
        Useful for precision allocation.
        """
        mask = [None] * seq_len
        # Sort by scale priority (Micro > Meso > Macro)
        sorted_caps = sorted(self.active_capsules.values(), key=lambda x: x.scale.value)
        
        for cap in sorted_caps:
            for i in range(max(0, cap.start_idx), min(seq_len, cap.end_idx + 1)):
                # Higher priority (Micro) overwrites lower
                mask[i] = cap.scale
        return mask
