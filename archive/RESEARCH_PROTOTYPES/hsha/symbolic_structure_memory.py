
from typing import Dict, List, Optional

class SymbolicStructureMemory:
    """
    PHASE 21.3: STRL - Symbolic Structure Memory.
    Stores historical topology snapshots and layout persistence.
    """
    def __init__(self):
        # iso_id -> list of historical token layouts (skeletons)
        self._history: Dict[str, List[List[int]]] = {}

    def record_snapshot(self, iso_id: str, tokens: List[int]):
        """Saves a snapshot of a successful symbolic layout."""
        if iso_id not in self._history:
            self._history[iso_id] = []
        # Keep only unique structural layouts
        if tokens not in self._history[iso_id]:
            self._history[iso_id].append(list(tokens))
            if len(self._history[iso_id]) > 5:
                self._history[iso_id].pop(0)

    def get_preferred_skeleton(self, iso_id: str) -> Optional[List[int]]:
        """Returns the most frequent or stable structural layout for an object."""
        if iso_id not in self._history:
            return None
        return self._history[iso_id][-1] # Return most recent stable one
