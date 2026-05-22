
from typing import List, Set, Dict, Any
import torch

class TopologyDriftDetector:
    """
    PHASE 21.3: STRL - Topology Drift Detector.
    Detects structural mutation and symbolic entropy spikes.
    """
    def __init__(self, delimiter_ids: Set[int]):
        self.delimiter_ids = delimiter_ids
        self.last_delimiter_pos = -1
        self.drift_score = 0.0

    def detect_drift(self, current_token_id: int, expected_topology: List[int], current_idx: int) -> float:
        """
        Calculates the drift score based on current token and expected topology.
        A score > 0.5 indicates significant structural deformation.
        """
        if not expected_topology:
            return 0.0
            
        # Check if the current token was expected to be a delimiter
        is_current_delimiter = current_token_id in self.delimiter_ids
        expected_delimiter = expected_topology[current_idx] if current_idx < len(expected_topology) else -1
        is_expected_delimiter = expected_delimiter in self.delimiter_ids
        
        # If we expected a delimiter but got something else, drift increases
        if is_expected_delimiter and not is_current_delimiter:
            self.drift_score = (self.drift_score * 0.7) + 0.3
        elif not is_expected_delimiter and is_current_delimiter:
            # Unexpected delimiter (structural pollution)
            self.drift_score = (self.drift_score * 0.7) + 0.2
        else:
            # Structural alignment maintained
            self.drift_score *= 0.8
            
        return self.drift_score

    def detect_entropy_spike(self, logits: torch.Tensor) -> bool:
        """Simple spike detection: is the top-1 probability significantly lower than usual?"""
        probs = torch.softmax(logits, dim=-1)
        top_p = torch.max(probs).item()
        return top_p < 0.1  # High uncertainty spike
