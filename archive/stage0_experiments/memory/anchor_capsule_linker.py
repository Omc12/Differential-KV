from typing import List, Dict
from .hierarchical_memory_capsules import MemoryCapsule
from .capsule_registry import CapsuleRegistry

class AnchorCapsuleLinker:
    """
    PHASE 18.7B: Anchor-Capsule Linker.
    Establishes pathways between semantic anchors (coarse) and HMCs (fine).
    """
    def __init__(self, registry: CapsuleRegistry):
        self.registry = registry
        self.anchor_to_capsules: Dict[int, List[str]] = {}

    def link_anchor_to_capsule(self, anchor_idx: int, capsule_id: str):
        if anchor_idx not in self.anchor_to_capsules:
            self.anchor_to_capsules[anchor_idx] = []
        
        if capsule_id not in self.anchor_to_capsules[anchor_idx]:
            self.anchor_to_capsules[anchor_idx].append(capsule_id)
            
            # Update the capsule's internal list
            if capsule_id in self.registry.capsules:
                self.registry.capsules[capsule_id].linked_anchors.append(anchor_idx)

    def get_capsules_for_anchor(self, anchor_idx: int) -> List[MemoryCapsule]:
        cids = self.anchor_to_capsules.get(anchor_idx, [])
        return [self.registry.capsules[cid] for cid in cids if cid in self.registry.capsules]

    def find_nearest_capsules(self, token_idx: int, radius: int = 128) -> List[MemoryCapsule]:
        """Finds capsules within a certain distance of a token (e.g. current query)."""
        nearby = []
        for capsule in self.registry.capsules.values():
            if abs(capsule.start_idx - token_idx) <= radius or abs(capsule.end_idx - token_idx) <= radius:
                nearby.append(capsule)
        return nearby
