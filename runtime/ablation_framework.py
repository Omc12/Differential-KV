"""
STAGE 2.5 - SRC: Ablation Framework
Phase 40.0 - Scientific Research Consolidation

Systematically disables systems to measure which components ACTUALLY matter.
"""
import threading
from typing import Dict, Any, List

class AblationFramework:
    def __init__(self):
        self._lock = threading.RLock()
        self.active_ablation = "none"
        self._ablation_results = {}
        
        self.ablation_states = [
            "none",
            "disable_adaptive_zoning",
            "disable_predictive_scheduling",
            "disable_semantic_repair",
            "disable_equilibrium_control"
        ]
        
    def set_ablation(self, state: str):
        with self._lock:
            if state in self.ablation_states:
                self.active_ablation = state
                if state not in self._ablation_results:
                    self._ablation_results[state] = {"tests": 0, "fidelity_sum": 0.0}
                    
    def record_outcome(self, fidelity: float):
        with self._lock:
            if self.active_ablation not in self._ablation_results:
                self._ablation_results[self.active_ablation] = {"tests": 0, "fidelity_sum": 0.0}
            
            self._ablation_results[self.active_ablation]["tests"] += 1
            self._ablation_results[self.active_ablation]["fidelity_sum"] += fidelity

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            res = {}
            for state, data in self._ablation_results.items():
                if data["tests"] > 0:
                    res[state] = round(data["fidelity_sum"] / data["tests"], 4)
                else:
                    res[state] = 0.0
            return {
                "active_ablation": self.active_ablation,
                "ablation_fidelities": res
            }
