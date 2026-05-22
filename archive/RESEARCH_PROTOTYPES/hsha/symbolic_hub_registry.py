
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class SymbolicObject:
    object_id: str
    tokens: List[int]
    topology_hash: str
    lineage: List[str] = field(default_factory=list) # List of parent object IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

class SymbolicHubRegistry:
    """
    PHASE 21.0: Immutable symbolic object storage & Hub registration.
    Part of the 'Post-Office System' for external symbolic persistence.
    """
    def __init__(self):
        self.hubs: Dict[str, SymbolicObject] = {}
        self.topology_to_id: Dict[str, str] = {}
        self.lineage_map: Dict[str, List[str]] = {}

    def register_hub(self, tokens: List[int], topology_hash: str, parent_ids: Optional[List[str]] = None) -> str:
        """
        Registers a symbolic object and returns its unique ID.
        Uses topology hashing for de-duplication and integrity.
        """
        # Create a unique ID based on content and topology
        token_str = ",".join(map(str, tokens))
        content_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]
        object_id = f"sym_{topology_hash}_{content_hash}"
        
        if object_id not in self.hubs:
            obj = SymbolicObject(
                object_id=object_id,
                tokens=tokens,
                topology_hash=topology_hash,
                lineage=parent_ids or []
            )
            self.hubs[object_id] = obj
            self.topology_to_id[topology_hash] = object_id
            
            # Track lineage
            if parent_ids:
                self.lineage_map[object_id] = parent_ids
                
        return object_id

    def register_root(self, start: int, tokens: List[int]):
        """
        Compatibility alias for SABEAF (20.8).
        Registers a symbolic sequence as a hub for future retrieval.
        """
        # For legacy 20.8 calls, we compute a quick topology hash from tokens
        token_str = ",".join(map(str, tokens))
        topology_hash = hashlib.sha256(token_str.encode()).hexdigest()[:12]
        self.register_hub(tokens, topology_hash)

    def get_object(self, object_id: str) -> Optional[SymbolicObject]:
        return self.hubs.get(object_id)

    def find_by_topology(self, topology_hash: str) -> Optional[SymbolicObject]:
        obj_id = self.topology_to_id.get(topology_hash)
        if obj_id:
            return self.hubs.get(obj_id)
        return None

    def get_lineage(self, object_id: str) -> List[str]:
        return self.lineage_map.get(object_id, [])

    def clear(self):
        self.hubs.clear()
        self.topology_to_id.clear()
        self.lineage_map.clear()
