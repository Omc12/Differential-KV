"""
STAGE 2 - OSE Hardening: Policy Circularity Trace
Phase 39.7 - Objective Semantic Evaluation

Traces circular governance amplification where policy affects metric,
metric reinforces policy, confidence rises, but external fidelity falls.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any
import threading

class PolicyCircularityTrace:
    def __init__(self, run_id: str):
        self._lock = threading.RLock()
        self.trace_dir = Path("traces/stage2/phase_39_7_ose") / run_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self.trace_dir / "policy_circularity_trace.jsonl", "a", encoding="utf-8", buffering=1)

    def record_circularity(self, step: int, best_policy: str, confidence: float, eq_score: float, fidelity: float):
        with self._lock:
            # Detect circular reinforcement
            is_circular = 1 if (confidence > 0.8 and eq_score > 0.9 and fidelity < 0.6) else 0
            
            data = {
                "ts": time.time(),
                "step": step,
                "policy": best_policy,
                "confidence": round(confidence, 4),
                "eq_score": round(eq_score, 4),
                "fidelity": round(fidelity, 4),
                "circularity_detected": is_circular
            }
            self._file.write(json.dumps(data) + "\n")

    def close(self):
        with self._lock:
            self._file.close()
