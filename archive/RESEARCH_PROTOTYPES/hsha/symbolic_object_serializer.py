
import json
from typing import Dict, Any
from .immutable_symbolic_object import ImmutableSymbolicObject

class SymbolicObjectSerializer:
    """
    PHASE 21.2: ISO - Compact object serialization.
    Handles safe transport and persistence encoding for ISOs.
    """
    @staticmethod
    def serialize(obj: ImmutableSymbolicObject) -> str:
        """Encodes an ISO into a compact JSON string."""
        data = {
            "id": obj.object_id,
            "tokens": obj.tokens,
            "t_hash": obj.topology_hash,
            "meta": obj.metadata
        }
        return json.dumps(data)

    @staticmethod
    def deserialize(data_str: str) -> ImmutableSymbolicObject:
        """Restores an ISO from a serialized string."""
        data = json.loads(data_str)
        return ImmutableSymbolicObject(
            tokens=data["tokens"],
            topology_hash=data["t_hash"],
            metadata=data["meta"]
        )
