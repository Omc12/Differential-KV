
from typing import Dict, List, Set, Optional, Any

class SymbolicRelationshipGraph:
    """
    PHASE 21.5: MHSR - Symbolic Relationship Graph.
    Tracks dependencies and associations between symbolic entities.
    """
    def __init__(self):
        # object_id -> set of related object_ids
        self._edges: Dict[str, Set[str]] = {}
        # relationship metadata (e.g., dependency, affinity)
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add_relationship(self, source_id: str, target_id: str, rel_type: str = "association"):
        """Adds a directional relationship between two symbolic objects."""
        if source_id not in self._edges:
            self._edges[source_id] = set()
        self._edges[source_id].add(target_id)
        
        rel_key = f"{source_id}->{target_id}"
        self._metadata[rel_key] = {"type": rel_type, "strength": 1.0}

    def get_related(self, object_id: str) -> List[str]:
        """Returns all object IDs related to the given object."""
        return list(self._edges.get(object_id, set()))

    def get_relationship_metadata(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(f"{source_id}->{target_id}")

    def clear(self):
        self._edges.clear()
        self._metadata.clear()
