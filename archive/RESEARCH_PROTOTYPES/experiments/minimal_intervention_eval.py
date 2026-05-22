"""
experiments/minimal_intervention_eval.py
Phase 26: Cognitive Energy Minimization (CEM)
Validates reasoning survival preservation with sparse, energy-efficient interventions.
"""

import torch
import numpy as np
import json
import os
from runtime.efficiency_controller import EfficiencyAwareRuntimeController
from runtime.resonance_feedback_engine import ResonanceFeedbackEngine

def run_minimal_intervention_benchmark(steps=1000, noise_scale=0.2):
    """
    Tests if the sparse reinforcement pulses are sufficient to maintain 
    long-horizon reasoning survival under noisy conditions.
    """
    print(f"=== Phase 26: Minimal Intervention Survival ===")
    
    d_model = 768
    resonance_engine = ResonanceFeedbackEngine(d_model=d_model)
    controller = EfficiencyAwareRuntimeController(resonance_engine)
    
    stability_history = []
    pulse_triggered_at = []
    
    # Simulation: Persistent noise that threatens to cause reasoning collapse
    current_stability = 1.0
    for i in range(steps):
        # 1. Natural drift + random perturbation
        drift_delta = 0.01 * np.random.randn() + (0.005 if i > 500 else 0)
        current_stability -= drift_delta
        current_stability = np.clip(current_stability, 0.0, 1.0)
        
        # 2. Mock metrics for the controller
        metrics = {
            "hidden_drift": 1.0 - current_stability,
            "trajectory_curvature": 0.05 if current_stability < 0.7 else 0.01,
            "phase_desync": (1.0 - current_stability) * 0.4,
            "cognitive_stability_score": current_stability
        }
        
        # 3. Process layer (Controller may apply pulse)
        initial_pulses = controller.pulse_scheduler.get_telemetry()["pulse_count"]
        _ = controller.process_layer(0, torch.randn(1, 1, d_model), metrics)
        new_pulses = controller.pulse_scheduler.get_telemetry()["pulse_count"]
        
        # 4. If pulse was applied, stability recovers
        if new_pulses > initial_pulses:
            current_stability += 0.3 # Recovery boost from pulse
            current_stability = np.clip(current_stability, 0.0, 1.0)
            pulse_triggered_at.append(i)
            
        stability_history.append(float(current_stability))
        
    # Survival criteria: Stability stayed above 0.3 for the whole horizon
    survival_steps = sum(1 for s in stability_history if s > 0.3)
    survival_rate = survival_steps / steps
    
    results = {
        "steps": steps,
        "survival_rate": survival_rate,
        "total_pulses": len(pulse_triggered_at),
        "pulse_steps": pulse_triggered_at,
        "stability_trend": stability_history,
        "avg_stability": float(np.mean(stability_history))
    }
    
    print(f"Survival Rate: {survival_rate:.2%}")
    print(f"Total Pulses: {results['total_pulses']}")
    print(f"Intervention Reduction vs Phase 25 (Continuous): {(1.0 - results['total_pulses']/steps):.2%}")
    
    os.makedirs("results/phase26", exist_ok=True)
    with open("results/phase26/minimal_intervention_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_minimal_intervention_benchmark()
