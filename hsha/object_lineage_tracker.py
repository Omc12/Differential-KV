
from typing import Dict, List, Optional
from .immutable_symbolic_object import ImmutableSymbolicObject

class ObjectLineageTracker:
    """
    PHASE 21.2: ISO - Symbolic ancestry tracking.
    Detects mutations and maintains lineage continuity.
    """
    def __init__(self):
        # child_id -> parent_id
        self._ancestry: Dict[str, str] = {}
        # parent_id -> list of child_ids
        self._descendants: Dict[str, List[str]] = {}

    def record_derivation(self, child: ImmutableSymbolicObject, parent: Optional[ImmutableSymbolicObject]):
        """Records that a child object was derived (recalled/mutated) from a parent."""
        if not parent:
            return
            
        p_id = parent.object_id
        c_id = child.object_id
        
        self._ancestry[c_id] = p_id
        if p_id not in self._descendants:
            self._descendants[p_id] = []
        if c_id not in self._descendants[p_id]:
            self._descendants[p_id].append(c_id)

    def detect_mutation(self, current_tokens: List[int], original_obj: ImmutableSymbolicObject) -> float:
        """
        Calculates mutation score (0.0 = exact match, 1.0 = completely different).
        Based on edit distance or simple token comparison.
        """
        orig_tokens = original_obj.tokens
        if len(current_tokens) != len(orig_tokens):
            return 1.0
            
        mismatches = sum(1 for a, b in zip(current_tokens, orig_tokens) if a != b)
        return mismatches / len(orig_tokens)

    def get_ancestry_chain(self, object_id: str) -> List[str]:
        chain = [object_id]
        curr = object_id
        while curr in self._ancestry:
            curr = self._ancestry[curr]
            chain.append(curr)
        return chain
