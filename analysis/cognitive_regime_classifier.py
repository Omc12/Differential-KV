"""
analysis/cognitive_regime_classifier.py
Phase 27: Adaptive Cognitive Routing (ACR)
Classifies the reasoning regime based on latent manifold dynamics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional

class CognitiveRegimeClassifier:
    def __init__(self):
        # Define regimes
        self.regimes = [
            "mathematical_reasoning",
            "code_generation",
            "recursive_planning",
            "tool_use_chains",
            "retrieval_heavy",
            "narrative_dialogue",
            "mixed_mode"
        ]
        
    def classify(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the current reasoning regime based on provided metrics.
        
        Metrics expected:
        - latent_drift: float
        - curvature: float
        - entropy_growth: float
        - resonance_coherence: float
        - branch_factor: float
        - attention_fragmentation: float
        - recursion_depth: int
        - token_acceleration: float
        """
        
        # Heuristic-based classification for Phase 27 initial implementation
        # In a real system, this would be a trained small MLP or decision forest
        
        scores = {regime: 0.0 for regime in self.regimes}
        
        drift = metrics.get("latent_drift", 0.0)
        curvature = metrics.get("curvature", 0.0)
        entropy = metrics.get("entropy_growth", 0.0)
        coherence = metrics.get("resonance_coherence", 1.0)
        branching = metrics.get("branch_factor", 1.0)
        fragmentation = metrics.get("attention_fragmentation", 0.0)
        depth = metrics.get("recursion_depth", 0)
        acceleration = metrics.get("token_acceleration", 0.0)
        
        # 1. Math: High curvature, low entropy (rigid), high coherence
        scores["mathematical_reasoning"] = (curvature * 0.4 + (1 - entropy) * 0.3 + coherence * 0.3)
        
        # 2. Code: Medium curvature, high fragmentation (local structures), medium drift
        scores["code_generation"] = (fragmentation * 0.4 + (1 - abs(curvature - 0.5)) * 0.3 + (1 - drift) * 0.3)
        
        # 3. Recursive Planning: High depth, high branching, high coherence
        scores["recursive_planning"] = (min(depth / 10.0, 1.0) * 0.4 + min(branching / 5.0, 1.0) * 0.3 + coherence * 0.3)
        
        # 4. Tool-use: High acceleration (state changes), high fragmentation
        scores["tool_use_chains"] = (acceleration * 0.5 + fragmentation * 0.3 + drift * 0.2)
        
        # 5. Retrieval: Low drift, high coherence, low curvature (linear retrieval)
        scores["retrieval_heavy"] = ((1 - drift) * 0.4 + coherence * 0.4 + (1 - curvature) * 0.2)
        
        # 6. Dialogue: High entropy, medium drift, low curvature
        scores["narrative_dialogue"] = (entropy * 0.5 + drift * 0.3 + (1 - curvature) * 0.2)
        
        # Normalize scores
        total = sum(scores.values()) + 1e-9
        probs = {k: v / total for k, v in scores.items()}
        
        # Predicted Instability Horizon
        # Higher drift and entropy = shorter horizon
        instability_horizon = 1.0 / (drift + entropy + 1e-9)
        
        # Expected Collapse Mode
        if drift > 0.8:
            collapse_mode = "trajectory_divergence"
        elif entropy > 0.8:
            collapse_mode = "semantic_washout"
        elif curvature > 0.8:
            collapse_mode = "geometry_shattering"
        else:
            collapse_mode = "gradual_decay"
            
        # Optimal Stabilization Strategy
        best_regime = max(probs, key=probs.get)
        strategy = self._get_optimal_strategy(best_regime)
        
        return {
            "regime_probabilities": probs,
            "best_regime": best_regime,
            "predicted_instability_horizon": instability_horizon,
            "expected_collapse_mode": collapse_mode,
            "optimal_stabilization_strategy": strategy
        }
        
    def _get_optimal_strategy(self, regime: str) -> str:
        strategies = {
            "mathematical_reasoning": "high_rigidity_lock",
            "code_generation": "local_correction_resonance",
            "recursive_planning": "persistent_attractor_reinforcement",
            "tool_use_chains": "high_frequency_pulse",
            "retrieval_heavy": "semantic_sync",
            "narrative_dialogue": "low_energy_adaptive",
            "mixed_mode": "balanced_hybrid"
        }
        return strategies.get(regime, "balanced_hybrid")

if __name__ == "__main__":
    classifier = CognitiveRegimeClassifier()
    sample_metrics = {
        "latent_drift": 0.1,
        "curvature": 0.8,
        "entropy_growth": 0.05,
        "resonance_coherence": 0.9,
        "branch_factor": 1.2,
        "attention_fragmentation": 0.2,
        "recursion_depth": 2,
        "token_acceleration": 0.1
    }
    result = classifier.classify(sample_metrics)
    print(f"Classification Result: {result['best_regime']} ({result['regime_probabilities'][result['best_regime']]:.2f})")
    print(f"Strategy: {result['optimal_stabilization_strategy']}")
