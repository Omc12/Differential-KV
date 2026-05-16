
import torch
from typing import Dict, Any, List, Optional

class SymbolicImportanceEstimator:
    """
    PHASE 23.4: CRS - Symbolic Importance Estimator.
    Estimates cognitive value and lineage criticality for residency prioritization.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.metrics = {
            "symbolic_priority_integrity": 1.0,
            "mean_symbolic_value": 0.0,
            "lineage_criticality_index": 0.0
        }

    def estimate_importance(self, 
                            hub_id: Optional[str], 
                            activation_frequency: float,
                            lineage_depth: int) -> float:
        """
        Calculates a priority score for a symbolic region.
        """
        # Base value from active hub presence
        hub_bonus = 0.5 if hub_id else 0.0
        
        # Frequency bonus (recency/repetition)
        freq_bonus = min(0.3, activation_frequency * 0.1)
        
        # Lineage criticality: deeper lineage often means more structural importance
        lineage_bonus = min(0.2, lineage_depth * 0.02)
        
        importance = hub_bonus + freq_bonus + lineage_bonus
        
        self.metrics["mean_symbolic_value"] = 0.9 * self.metrics["mean_symbolic_value"] + 0.1 * importance
        self.metrics["lineage_criticality_index"] = lineage_bonus
        
        return min(1.0, importance)

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
