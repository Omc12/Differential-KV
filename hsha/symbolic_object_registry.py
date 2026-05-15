
from typing import Dict, Optional, List
from .immutable_symbolic_object import ImmutableSymbolicObject

class SymbolicObjectRegistry:
    """
    PHASE 21.2: ISO - Persistent symbolic object registry.
    Manages the lifecycle and identity of immutable symbolic entities.
    """
    def __init__(self):
        self._objects: Dict[str, ImmutableSymbolicObject] = {}
        self._topology_index: Dict[str, List[str]] = {}  # topology_hash -> list of object_ids

    def register_object(self, obj: ImmutableSymbolicObject):
        """Registers an immutable object in the system."""
        self._objects[obj.object_id] = obj
        
        t_hash = obj.topology_hash
        if t_hash not in self._topology_index:
            self._topology_index[t_hash] = []
        if obj.object_id not in self._topology_index[t_hash]:
            self._topology_index[t_hash].append(obj.object_id)

    def get_object(self, object_id: str) -> Optional[ImmutableSymbolicObject]:
        return self._objects.get(object_id)

    def find_by_topology(self, topology_hash: str) -> List[ImmutableSymbolicObject]:
        """Finds all objects sharing the same structural topology."""
        ids = self._topology_index.get(topology_hash, [])
        return [self._objects[oid] for oid in ids]

    def list_all(self) -> List[str]:
        return list(self._objects.keys())

    def clear(self):
        self._objects.clear()
        self._topology_index.clear()
