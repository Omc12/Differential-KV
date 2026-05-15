
from typing import Dict, List, Optional
from .immutable_symbolic_object import ImmutableSymbolicObject

class TemporalLineageTracker:
    """
    PHASE 21.4: LSCP - Temporal Lineage Tracker.
    Maintains symbolic ancestry across dormancy and session transitions.
    """
    def __init__(self):
        # child_id -> (parent_id, session_id)
        self._temporal_ancestry: Dict[str, tuple] = {}

    def record_resurrection(self, obj: ImmutableSymbolicObject, session_id: str):
        """Records that an object was resurrected in a specific session."""
        o_id = obj.object_id
        if o_id not in self._temporal_ancestry:
            # First time resurrection in this tracker's scope
            self._temporal_ancestry[o_id] = (None, session_id)
        else:
            # Already exists, just update session context if needed
            pass

    def get_lineage_depth(self, object_id: str) -> int:
        depth = 0
        curr = object_id
        while curr in self._temporal_ancestry and self._temporal_ancestry[curr][0]:
            curr = self._temporal_ancestry[curr][0]
            depth += 1
        return depth
