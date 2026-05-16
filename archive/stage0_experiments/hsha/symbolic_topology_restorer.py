
from typing import List, Optional, Set
from .topology_drift_detector import TopologyDriftDetector
from .delimiter_integrity_guard import DelimiterIntegrityGuard
from .probabilistic_topology_blender import ProbabilisticTopologyBlender
from .symbolic_structure_memory import SymbolicStructureMemory

class SymbolicTopologyRestorer:
    """
    PHASE 21.3: STRL - Symbolic Topology Restorer.
    Main engine for detecting and healing symbolic structure drift.
    """
    def __init__(self, delimiter_ids: Set[int]):
        self.drift_detector = TopologyDriftDetector(delimiter_ids)
        self.integrity_guard = DelimiterIntegrityGuard(delimiter_ids)
        self.blender = ProbabilisticTopologyBlender()
        self.structure_memory = SymbolicStructureMemory()
        
        self.active_iso_id: Optional[str] = None
        self.expected_skeleton: List[int] = []

    def prepare_restoration(self, iso_id: str, tokens: List[int]):
        """Sets the target object for restoration tracking."""
        self.active_iso_id = iso_id
        self.expected_skeleton = tokens
        self.structure_memory.record_snapshot(iso_id, tokens)

    def process_token(self, token_id: int, current_idx: int):
        """Updates internal state with the latest generated token."""
        if not self.active_iso_id:
            return
            
        is_expected_structural = False
        if current_idx < len(self.expected_skeleton):
            # Check if we expected a delimiter at this position
            if self.expected_skeleton[current_idx] in self.drift_detector.delimiter_ids:
                is_expected_structural = True
                
        self.drift_detector.detect_drift(token_id, self.expected_skeleton, current_idx)
        self.integrity_guard.record_token(token_id, is_expected_structural)

    def heal_topology(self, logits, current_idx):
        """Applies self-healing logic to the logits if drift is detected."""
        if not self.active_iso_id or current_idx >= len(self.expected_skeleton):
            return logits
            
        drift = self.drift_detector.drift_score
        stabilization = self.integrity_guard.get_stabilization_bias()
        
        # Combined repair signal
        repair_signal = max(drift, stabilization)
        
        if repair_signal > 0.3:
            # Targeted repair: we know exactly what token should be here in the topology
            target_token = self.expected_skeleton[current_idx]
            # Probabilistic blending
            logits = self.blender.blend_repair(logits, {target_token}, repair_signal)
            
        return logits
