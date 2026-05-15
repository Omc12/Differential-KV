
from typing import List, Set

class TraversalLegitimacyGuard:
    """
    PHASE 21.5: MHSR - Traversal Legitimacy Guard.
    Prevents recursion and validates multi-hop symbolic chains.
    """
    def __init__(self, max_hops: int = 3):
        self.max_hops = max_hops
        self.active_chain: List[str] = []

    def can_traverse(self, target_id: str) -> bool:
        """
        Checks if the traversal to target_id is legitimate.
        Prevents recursion and over-extended chains.
        """
        # 1. Check for recursion (cycle detection)
        if target_id in self.active_chain:
            # print(f"[DEBUG] MHSR: Recursion detected for {target_id}")
            return False
            
        # 2. Check chain length
        if len(self.active_chain) >= self.max_hops:
            # print(f"[DEBUG] MHSR: Max hops reached ({self.max_hops})")
            return False
            
        return True

    def enter_hop(self, object_id: str):
        self.active_chain.append(object_id)

    def exit_hop(self):
        if self.active_chain:
            self.active_chain.pop()

    def reset(self):
        self.active_chain = []
