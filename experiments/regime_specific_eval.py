"""
experiments/regime_specific_eval.py
Phase 27: Adaptive Cognitive Routing (ACR)
Evaluates reasoning survival and efficiency across specific cognitive regimes.
"""

import os
import torch
import numpy as np
import json
from typing import Dict, List, Any
from runtime.cognitive_policy_engine import CognitivePolicyEngine

class RegimeSpecificEvaluator:
    def __init__(self, model_id="Qwen/Qwen2-0.5B"):
        self.engine = CognitivePolicyEngine(num_layers=24)
        self.results = {}
        
    def simulate_regime_trajectory(self, regime_name: str, num_steps: int = 100) -> Dict[str, Any]:
        """
        Simulates a trajectory for a specific regime with characteristic metrics.
        """
        history = []
        survival = 1.0
        total_pulses = 0
        total_overhead = 0
        
        # Characteristic metrics per regime
        base_metrics = {
            "mathematical_reasoning": {"drift": 0.05, "curvature": 0.8, "entropy": 0.05, "coherence": 0.9, "depth": 2},
            "code_generation": {"drift": 0.1, "curvature": 0.5, "entropy": 0.1, "coherence": 0.7, "depth": 1},
            "recursive_planning": {"drift": 0.15, "curvature": 0.6, "entropy": 0.2, "coherence": 0.8, "depth": 10},
            "retrieval_heavy": {"drift": 0.02, "curvature": 0.1, "entropy": 0.05, "coherence": 0.95, "depth": 0},
            "narrative_dialogue": {"drift": 0.3, "curvature": 0.2, "entropy": 0.6, "coherence": 0.5, "depth": 0}
        }
        
        chars = base_metrics.get(regime_name, base_metrics["narrative_dialogue"])
        
        for i in range(num_steps):
            # Add some noise and drift growth
            step_metrics = {
                "latent_drift": chars["drift"] * (1 + 0.005 * i) + np.random.normal(0, 0.01),
                "curvature": chars["curvature"] + np.random.normal(0, 0.05),
                "entropy_growth": chars["entropy"] + np.random.normal(0, 0.02),
                "resonance_coherence": chars["coherence"] - (0.001 * i),
                "branch_factor": 1.1,
                "attention_fragmentation": 0.1,
                "recursion_depth": chars["depth"],
                "token_acceleration": 0.05
            }
            
            # Policy Engine Intervention
            state = self.engine.step(step_metrics)
            
            # Metrics tracking
            pulse_freq = state["resonance"]["pulse_frequency"]
            total_pulses += 1 if np.random.random() < pulse_freq else 0
            total_overhead += state["budget"]["estimated_overhead"]
            
            # Survival logic: if drift exceeds threshold, survival drops
            if step_metrics["latent_drift"] > 1.0:
                survival *= 0.9
            
            history.append({
                "step": i,
                "metrics": step_metrics,
                "state": state,
                "survival": survival
            })
            
        return {
            "regime": regime_name,
            "avg_survival": survival,
            "pulse_density": total_pulses / num_steps,
            "avg_overhead": total_overhead / num_steps,
            "final_drift": history[-1]["metrics"]["latent_drift"],
            "history": history
        }

    def run_suite(self):
        regimes = ["mathematical_reasoning", "code_generation", "recursive_planning", "retrieval_heavy", "narrative_dialogue"]
        for r in regimes:
            print(f"Evaluating regime: {r}...")
            self.results[r] = self.simulate_regime_trajectory(r)
            print(f"  Survival: {self.results[r]['avg_survival']:.2f}, Pulses: {self.results[r]['pulse_density']:.4f}")
            
        os.makedirs("results/phase27", exist_ok=True)
        with open("results/phase27/regime_specific_results.json", "w") as f:
            # Clean history for JSON
            json_results = {k: {sk: sv for sk, sv in v.items() if sk != "history"} for k, v in self.results.items()}
            json.dump(json_results, f, indent=4)

if __name__ == "__main__":
    evaluator = RegimeSpecificEvaluator()
    evaluator.run_suite()
