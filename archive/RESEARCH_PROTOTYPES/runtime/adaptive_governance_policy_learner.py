"""
STAGE 2 - ASI: Adaptive Governance Policy Learner
Phase 39.6 - Adaptive Semantic Intelligence

Learns which governance actions produce the best long-term semantic stability.
Provides policy suggestions without overriding safety guards.
"""
import threading
from typing import Dict, Any, List

class AdaptiveGovernancePolicyLearner:
    def __init__(self):
        self._lock = threading.RLock()
        
        # Policy -> confidence score (0.0 to 1.0)
        self._policy_confidence: Dict[str, float] = {
            "early_densification": 0.5,
            "delayed_repair": 0.5,
            "aggressive_hybrid": 0.5,
            "anchor_heavy": 0.5
        }
        self._policy_samples: Dict[str, int] = {k: 0 for k in self._policy_confidence.keys()}

    def record_policy_outcome(self, policy: str, stability_duration: int, drift_reduction: float):
        """Learn from the outcome of applying a specific policy."""
        with self._lock:
            if policy not in self._policy_confidence:
                return
                
            # Score based on how long stability lasted and how much drift reduced
            outcome_score = min(1.0, (stability_duration / 50.0) * 0.5 + (drift_reduction / 5.0) * 0.5)
            
            current_conf = self._policy_confidence[policy]
            samples = self._policy_samples[policy]
            
            # Moving average
            self._policy_confidence[policy] = (current_conf * samples + outcome_score) / (samples + 1)
            self._policy_samples[policy] += 1

    def suggest_best_policy(self) -> str:
        """Returns the policy with the highest learned confidence."""
        with self._lock:
            return max(self._policy_confidence.items(), key=lambda x: x[1])[0]

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            best_policy = self.suggest_best_policy()
            return {
                "top_policy": best_policy,
                "top_policy_confidence": round(self._policy_confidence[best_policy], 4),
                "total_learning_samples": sum(self._policy_samples.values())
            }
