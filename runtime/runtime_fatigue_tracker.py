import numpy as np
from typing import List, Dict, Any

class RuntimeFatigueTracker:
    """
    Monitors performance degradation and drift over long periods.
    """
    def __init__(self):
        self.history = []

    def log_step(self, metrics: Dict[str, Any]):
        self.history.append(metrics)

    def generate_report(self) -> Dict[str, Any]:
        if not self.history:
            return {}
            
        tps_values = [h['tps'] for h in self.history]
        start_tps = tps_values[0]
        end_tps = tps_values[-1]
        drift = (end_tps - start_tps) / start_tps
        
        return {
            "total_steps": len(self.history),
            "avg_tps": np.mean(tps_values),
            "tps_drift": drift,
            "max_latency": max(h['latency'] for h in self.history),
            "fatigue_score": abs(drift) * 100 # Higher is worse
        }
