from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch

@dataclass
class MemoryCapsule:
    """
    PHASE 18.7A: Hierarchical Memory Capsule (HMC).
    Represents a protected, high-fidelity symbolic memory region linked to semantic anchors.
    """
    capsule_id: str
    start_idx: int
    end_idx: int
    precision_tier: str # 'HIGH' (Symbolic), 'MEDIUM' (Semantic), 'LOW' (Filler)
    entropy_score: float
    linked_anchors: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    kv_states: Optional[Dict[int, torch.Tensor]] = None # layer_idx -> tensor
    last_accessed: int = 0 # timestamp or step
    activation_count: int = 0

    def __repr__(self):
        return f"HMC({self.capsule_id}, [{self.start_idx}:{self.end_idx}], tier={self.precision_tier}, entropy={self.entropy_score:.4f})"

class HierarchicalMemoryCapsuleEngine:
    """
    Manages the creation and logical grouping of capsules.
    """
    def __init__(self, high_fidelity_budget: int = 1024):
        self.budget = high_fidelity_budget
        self.capsules: Dict[str, MemoryCapsule] = {}
        self.current_step = 0

    def create_capsule(self, start: int, end: int, tier: str, entropy: float, anchors: List[int] = None, tags: List[str] = None) -> MemoryCapsule:
        cid = f"hmc_{start}_{end}_{tier.lower()}"
        capsule = MemoryCapsule(
            capsule_id=cid,
            start_idx=start,
            end_idx=end,
            precision_tier=tier,
            entropy_score=entropy,
            linked_anchors=anchors or [],
            tags=tags or [],
            last_accessed=self.current_step
        )
        self.capsules[cid] = capsule
        return capsule

    def get_active_capsules(self) -> List[MemoryCapsule]:
        return list(self.capsules.values())

    def update_step(self):
        self.current_step += 1
