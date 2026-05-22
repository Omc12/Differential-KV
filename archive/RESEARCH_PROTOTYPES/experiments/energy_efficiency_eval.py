"""
experiments/energy_efficiency_eval.py
Phase 26: Cognitive Energy Minimization (CEM)
Evaluates cognitive energy reduction and stability basin transitions.
"""

import torch
import numpy as np
import json
import os
from runtime.efficiency_controller import EfficiencyAwareRuntimeController
from runtime.resonance_feedback_engine import ResonanceFeedbackEngine
from analysis.energy_landscape import EnergyLandscapeMapper

def run_energy_benchmark(steps=500):
    """
    Simulates a long-horizon reasoning trajectory and measures the 
    cognitive energy landscape.
    """
    print(f"=== Phase 26: Energy Efficiency Evaluation ===")
    
    d_model = 768
    resonance_engine = ResonanceFeedbackEngine(d_model=d_model)
    controller = EfficiencyAwareRuntimeController(resonance_engine)
    landscape_mapper = EnergyLandscapeMapper(stable_threshold=0.15, collapse_threshold=0.4)
    
    energy_history = []
    basin_history = []
    
    # Simulation loop: Stable -> Perturbed -> Stabilized -> Stable
    for i in range(steps):
        # Construct synthetic instability
        if i < 100: # Initial stability
            drift = 0.05 + 0.01 * np.random.rand()
        elif i < 200: # Rising instability
            drift = 0.05 + 0.003 * (i - 100)
        elif i < 300: # High instability/Pulse region
            drift = 0.35 + 0.05 * np.random.rand()
        else: # Post-pulse recovery
            drift = 0.15 - 0.0005 * (i - 300)
            
        metrics = {
            "hidden_drift": drift,
            "trajectory_curvature": 0.02 + 0.1 * (drift if drift > 0.3 else 0),
            "phase_desync": drift * 0.5,
            "cognitive_stability_score": 1.0 - drift
        }
        
        _ = controller.process_layer(0, torch.randn(1, 1, d_model), metrics)
        
        current_energy = controller.energy_model.get_history()[-1]
        energy_history.append(current_energy)
        
        landscape_mapper.update_trajectory(current_energy)
        basin_history.append(landscape_mapper.current_basin)
        
    results = {
        "avg_energy": float(np.mean(energy_history)),
        "max_energy": float(np.max(energy_history)),
        "energy_curve": energy_history,
        "basin_transitions": landscape_mapper.get_transition_probabilities(),
        "basin_distribution": landscape_mapper.get_basin_stats(),
        "pulse_frequency": controller.pulse_scheduler.get_pulse_frequency()
    }
    
    print(f"Average Energy: {results['avg_energy']:.4f}")
    print(f"Pulse Frequency: {results['pulse_frequency']:.2%}")
    print(f"Basin Distribution: {results['basin_distribution']}")
    
    os.makedirs("results/phase26", exist_ok=True)
    with open("results/phase26/energy_curves.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_energy_benchmark()
