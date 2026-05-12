"""
experiments/mixed_cognition_eval.py
Phase 27: Adaptive Cognitive Routing (ACR)
Evaluates transitions between different cognitive regimes.
"""

import os
import numpy as np
import json
from typing import Dict, List, Any
from runtime.cognitive_policy_engine import CognitivePolicyEngine

class MixedCognitionEvaluator:
    def __init__(self):
        self.engine = CognitivePolicyEngine(num_layers=24)
        
    def run_mixed_trajectory(self, sequence: List[str], steps_per_regime: int = 50) -> Dict[str, Any]:
        """
        Runs a sequence of regimes to test adaptation latency and stability.
        """
        history = []
        total_steps = len(sequence) * steps_per_regime
        
        base_metrics = {
            "mathematical_reasoning": {"drift": 0.05, "curvature": 0.8, "entropy": 0.05, "coherence": 0.9, "depth": 2},
            "code_generation": {"drift": 0.1, "curvature": 0.5, "entropy": 0.1, "coherence": 0.7, "depth": 1},
            "recursive_planning": {"drift": 0.15, "curvature": 0.6, "entropy": 0.2, "coherence": 0.8, "depth": 10},
            "retrieval_heavy": {"drift": 0.02, "curvature": 0.1, "entropy": 0.05, "coherence": 0.95, "depth": 0},
            "narrative_dialogue": {"drift": 0.3, "curvature": 0.2, "entropy": 0.6, "coherence": 0.5, "depth": 0}
        }
        
        current_step = 0
        for regime_name in sequence:
            chars = base_metrics[regime_name]
            for _ in range(steps_per_regime):
                step_metrics = {
                    "latent_drift": chars["drift"] + np.random.normal(0, 0.02),
                    "curvature": chars["curvature"] + np.random.normal(0, 0.05),
                    "entropy_growth": chars["entropy"] + np.random.normal(0, 0.02),
                    "resonance_coherence": chars["coherence"],
                    "branch_factor": 1.1,
                    "attention_fragmentation": 0.1,
                    "recursion_depth": chars["depth"],
                    "token_acceleration": 0.05
                }
                
                state = self.engine.step(step_metrics)
                
                history.append({
                    "step": current_step,
                    "target_regime": regime_name,
                    "detected_regime": state["regime"],
                    "resonance_freq": state["resonance"]["pulse_frequency"],
                    "overhead": state["budget"]["estimated_overhead"]
                })
                current_step += 1
                
        # Calculate transition accuracy
        accuracy = sum(1 for h in history if h["target_regime"] == h["detected_regime"]) / total_steps
        
        return {
            "sequence": sequence,
            "total_steps": total_steps,
            "regime_detection_accuracy": accuracy,
            "history": history
        }

if __name__ == "__main__":
    evaluator = MixedCognitionEvaluator()
    sequence = ["retrieval_heavy", "mathematical_reasoning", "code_generation", "narrative_dialogue", "recursive_planning"]
    result = evaluator.run_mixed_trajectory(sequence)
    print(f"Mixed Trajectory Accuracy: {result['regime_detection_accuracy']:.2%}")
    
    os.makedirs("results/phase27", exist_ok=True)
    with open("results/phase27/mixed_cognition_results.json", "w") as f:
        # Save a summary and sample of history
        summary = {
            "accuracy": result["regime_detection_accuracy"],
            "sequence": result["sequence"],
            "history_sample": result["history"][::10] # Every 10th step
        }
        json.dump(summary, f, indent=4)
