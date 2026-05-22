"""
STAGE 2 - ASS: Proactive Recovery Coordinator
Phase 39.5 - Adaptive Semantic Scheduling

Schedules targeted recovery actions BEFORE semantic collapse manifests.
Coordinates with the predictive pressure estimator and anchor analyzer.
"""
import threading
from typing import Dict, Any, List

class ProactiveRecoveryCoordinator:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._lock = threading.RLock()
        
        self._proactive_recoveries = 0

    def schedule_proactive_recoveries(self, step: int, pressure_map: Dict[int, float], anchor_analyzer: Any) -> List[int]:
        """
        Determines which layers need proactive dense reinforcement.
        """
        with self._lock:
            to_recover = []
            for i in range(self.num_layers):
                pressure = pressure_map.get(i, 0.0)
                anchor_failing = anchor_analyzer.will_fail_soon(step, i)
                
                # If pressure is moderately high AND anchor is about to fail,
                # preemptively recover instead of waiting for a drift spike.
                if pressure > 0.6 and anchor_failing:
                    to_recover.append(i)
                    self._proactive_recoveries += 1
                    
            return to_recover

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_proactive_recoveries": self._proactive_recoveries
            }
