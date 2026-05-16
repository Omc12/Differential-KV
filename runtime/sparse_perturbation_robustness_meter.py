"""
STAGE 2 - ARS: Sparse Perturbation Robustness Meter
Phase 39.8 - Adversarial Reasoning Stability

Measures how fragile reasoning becomes under sparse perturbation pressure.
"""
import threading
from typing import Dict, Any

class SparsePerturbationRobustnessMeter:
    def __init__(self):
        self._lock = threading.RLock()
        self._total_perturbations = 0
        self._survived_perturbations = 0

    def evaluate_robustness(self, sparse_logits_div: float, internal_drift: float):
        with self._lock:
            self._total_perturbations += 1
            # If KL divergence is low despite internal drift being present, it survived the perturbation
            if sparse_logits_div < 0.5:
                self._survived_perturbations += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self._total_perturbations, 1)
            return {
                "perturbation_robustness_score": round(self._survived_perturbations / total, 4)
            }
