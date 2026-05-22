
from typing import List, Dict, Optional, Set, Any

class TopologyIntegrityMap:
    """
    PHASE 21.0: Tracks the structural health of the symbolic stream.
    Maintains a 'skeleton' of delimiters to verify topology integrity.
    """
    def __init__(self, delimiter_ids: Set[int]):
        self.delimiter_ids = delimiter_ids
        self.active_skeleton: List[int] = []
        self.drift_events = 0
        self.total_tokens = 0
        self.expected_delimiter_pos: List[int] = [] # Relative positions

    def record_token(self, token_id: int, is_expected_delimiter: bool):
        """
        Tracks the current generation and detects structural drift.
        """
        self.total_tokens += 1
        if token_id in self.delimiter_ids:
            self.active_skeleton.append(token_id)
            if not is_expected_delimiter:
                # Unexpected delimiter or wrong position
                self.drift_events += 1
        elif is_expected_delimiter:
            # Missing expected delimiter
            self.drift_events += 1

    def get_drift_risk(self) -> float:
        """
        Returns a risk score [0, 1] representing structural instability.
        High risk triggers 'Soft Topology Restoration'.
        """
        if self.total_tokens == 0:
            return 0.0
        
        # Risk scales with drift events relative to skeleton size
        risk = (self.drift_events * 2.0) / (len(self.active_skeleton) + 1)
        return min(1.0, risk)

    def verify_topology(self, current_tokens: List[int], target_topology_hash: str, 
                        encoder: Any) -> bool:
        """
        Verifies if the current window matches the target topology.
        """
        current_hash = encoder.extract_topology(current_tokens)
        return current_hash == target_topology_hash

    def reset(self):
        self.active_skeleton.clear()
        self.drift_events = 0
        self.total_tokens = 0
