
from typing import Dict, List, Optional
import time
from .immutable_symbolic_object import ImmutableSymbolicObject

class DormantSymbolicRegistry:
    """
    PHASE 21.4: LSCP - Dormant Symbolic Registry.
    Manages symbolic entities that have exited active attention but persist in storage.
    """
    def __init__(self):
        self._dormant_objects: Dict[str, ImmutableSymbolicObject] = {}
        self._last_access: Dict[str, float] = {}

    def move_to_dormancy(self, obj: ImmutableSymbolicObject):
        """Archives an active object into dormant storage."""
        self._dormant_objects[obj.object_id] = obj
        self._last_access[obj.object_id] = time.time()

    def resurrect(self, object_id: str) -> Optional[ImmutableSymbolicObject]:
        """Retrieves and 'revives' a dormant object."""
        if object_id in self._dormant_objects:
            self._last_access[object_id] = time.time()
            return self._dormant_objects[object_id]
        return None

    def list_dormant(self) -> List[str]:
        return list(self._dormant_objects.keys())

    def purge_stale(self, max_age_seconds: float):
        """Removes objects that haven't been accessed for a long time."""
        now = time.time()
        to_purge = [oid for oid, last in self._last_access.items() if now - last > max_age_seconds]
        for oid in to_purge:
            del self._dormant_objects[oid]
            del self._last_access[oid]
        return len(to_purge)
