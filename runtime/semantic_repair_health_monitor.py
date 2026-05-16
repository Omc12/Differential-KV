from typing import Dict, Any, List
import time

class SemanticRepairHealthMonitor:
    """
    STAGE 2 - SRI: Semantic Repair Health Monitor
    Tracks the active participation and confidence of low-rank semantic repairs.
    """
    def __init__(self):
        self.stats = {
            "total_steps": 0,
            "repair_activations": 0,
            "avg_anchor_stability": 1.0,
            "avg_repair_confidence": 1.0
        }
        self._anchor_hist: List[float] = []
        self._repair_hist: List[float] = []
        
    def record_repair_step(self, activated: bool, anchor_stability: float, repair_confidence: float):
        self.stats["total_steps"] += 1
        if activated:
            self.stats["repair_activations"] += 1
            
        self._anchor_hist.append(anchor_stability)
        self._repair_hist.append(repair_confidence)
        
        # Keep recent history for moving average
        if len(self._anchor_hist) > 100: self._anchor_hist.pop(0)
        if len(self._repair_hist) > 100: self._repair_hist.pop(0)
        
        self.stats["avg_anchor_stability"] = sum(self._anchor_hist) / max(len(self._anchor_hist), 1)
        self.stats["avg_repair_confidence"] = sum(self._repair_hist) / max(len(self._repair_hist), 1)

    def get_health_metrics(self) -> Dict[str, Any]:
        return {
            "repair_activation_rate": self.stats["repair_activations"] / max(self.stats["total_steps"], 1),
            "anchor_stability": self.stats["avg_anchor_stability"],
            "repair_confidence": self.stats["avg_repair_confidence"]
        }
