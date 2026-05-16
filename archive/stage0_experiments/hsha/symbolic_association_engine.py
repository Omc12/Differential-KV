
from typing import Dict, List, Set
from .immutable_symbolic_object import ImmutableSymbolicObject

class SymbolicAssociationEngine:
    """
    PHASE 21.5: MHSR - Symbolic Association Engine.
    Models affinity and co-occurrence between symbolic entities.
    """
    def __init__(self):
        # pair -> co-occurrence count
        self._affinity_scores: Dict[str, float] = {}

    def record_co_occurrence(self, obj1_id: str, obj2_id: str):
        """Updates affinity between two objects observed in the same context."""
        pair = tuple(sorted([obj1_id, obj2_id]))
        key = f"{pair[0]}<->{pair[1]}"
        self._affinity_scores[key] = self._affinity_scores.get(key, 0.0) + 1.0

    def get_affinity(self, obj1_id: str, obj2_id: str) -> float:
        """Returns the normalized affinity score between two objects."""
        pair = tuple(sorted([obj1_id, obj2_id]))
        key = f"{pair[0]}<->{pair[1]}"
        score = self._affinity_scores.get(key, 0.0)
        # Simplified normalization for MVP
        return min(1.0, score / 10.0)
