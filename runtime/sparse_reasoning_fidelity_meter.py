"""
STAGE 2 - OSE: Sparse Reasoning Fidelity Meter
Phase 39.7 - Objective Semantic Evaluation

Estimates how much reasoning fidelity survives under sparse governance.
"""
import threading
from typing import Dict, Any

class SparseReasoningFidelityMeter:
    def __init__(self):
        self._lock = threading.RLock()
        self._fidelity_score = 1.0 # 1.0 = perfect fidelity

    def update_fidelity(self, exact_match_rate: float, agreement_rate: float, kl_div: float):
        with self._lock:
            # Fidelity drops if exact matches fall, reasoning diverges, or KL div spikes
            match_component = exact_match_rate * 0.4
            agree_component = agreement_rate * 0.4
            div_component = max(0.0, 1.0 - kl_div / 2.0) * 0.2
            
            new_fidelity = match_component + agree_component + div_component
            
            # Smooth updates
            self._fidelity_score = (self._fidelity_score * 0.8) + (new_fidelity * 0.2)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "fidelity_score": round(self._fidelity_score, 4)
            }
