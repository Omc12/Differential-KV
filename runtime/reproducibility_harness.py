"""
STAGE 2.5 - SRC: Reproducibility Harness
Phase 40.0 - Scientific Research Consolidation

Verifies stability across repeated runs and random seeds.
"""
import threading
from typing import Dict, Any

class ReproducibilityHarness:
    def __init__(self):
        self._lock = threading.RLock()
        self._run_fidelities = []

    def record_run(self, fidelity: float):
        with self._lock:
            self._run_fidelities.append(fidelity)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            if len(self._run_fidelities) < 2:
                return {"variance": 0.0, "runs": len(self._run_fidelities)}
            
            mean = sum(self._run_fidelities) / len(self._run_fidelities)
            variance = sum((x - mean) ** 2 for x in self._run_fidelities) / len(self._run_fidelities)
            return {
                "variance": round(variance, 6),
                "runs": len(self._run_fidelities)
            }
