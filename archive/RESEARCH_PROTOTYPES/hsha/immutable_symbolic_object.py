
import hashlib
from typing import List, Dict, Any, Optional

class ImmutableSymbolicObject:
    """
    PHASE 21.2: ISO - Immutable Symbolic Object.
    Represents a symbolic entity with stable identity and topology.
    Payload is immutable once created.
    """
    def __init__(self, tokens: List[int], topology_hash: str, metadata: Optional[Dict[str, Any]] = None):
        self._tokens = tuple(tokens)  # Ensure immutability
        self._topology_hash = topology_hash
        self._metadata = metadata or {}
        
        # Identity is derived from payload and topology
        payload_hash = hashlib.sha256(str(self._tokens).encode()).hexdigest()[:16]
        self._object_id = f"iso_{topology_hash[:8]}_{payload_hash}"
        
        self._created_at = metadata.get("timestamp", 0)
        self._source_pos = metadata.get("source_pos", -1)

    @property
    def object_id(self) -> str:
        return self._object_id

    @property
    def tokens(self) -> List[int]:
        return list(self._tokens)

    @property
    def topology_hash(self) -> str:
        return self._topology_hash

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self) -> str:
        return f"ImmutableSymbolicObject(id={self._object_id}, len={len(self)})"
