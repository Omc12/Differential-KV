"""
experiments/throughput_recovery_eval.py
Phase 26: Cognitive Energy Minimization (CEM)
Measures token/sec improvements and throughput recovery by reducing intervention density.
"""

import time
import torch
import numpy as np
import json
import os
from typing import Dict, List
from runtime.efficiency_controller import EfficiencyAwareRuntimeController
from runtime.resonance_feedback_engine import ResonanceFeedbackEngine

def run_throughput_benchmark(steps=250, compression_ratio=16.0):
    """
    Benchmarks the throughput of the efficiency-aware runtime compared to 
    continuous stabilization baselines.
    """
    print(f"=== Phase 26: Throughput Recovery Benchmark ===")
    print(f"Config: {steps} steps, {compression_ratio}x compression")
    
    d_model = 768
    # 1. Baseline: Continuous Reinforcement (Phase 25 style)
    resonance_engine_base = ResonanceFeedbackEngine(d_model=d_model, reinforcement_interval=1)
    
    # 2. Optimized: Efficiency-Aware Sparse Reinforcement (Phase 26)
    resonance_engine_opt = ResonanceFeedbackEngine(d_model=d_model, reinforcement_interval=1)
    controller = EfficiencyAwareRuntimeController(resonance_engine_opt)
    
    # Run Baseline
    start_base = time.time()
    for i in range(steps):
        latent = torch.randn(1, 1, d_model)
        # Phase 25 update (simulated)
        _ = resonance_engine_base.update_resonance(0, latent)
    end_base = time.time()
    
    # Run Optimized
    start_opt = time.time()
    for i in range(steps):
        latent = torch.randn(1, 1, d_model)
        # Mock metrics that fluctuate
        metrics = {
            "hidden_drift": 0.05 + 0.01 * np.sin(i / 10.0),
            "trajectory_curvature": 0.02,
            "phase_desync": 0.05,
            "cognitive_stability_score": 0.98 - 0.01 * (i / 100.0)
        }
        # Phase 26 efficiency-aware process
        _ = controller.process_layer(0, latent, metrics)
    end_opt = time.time()
    
    t_base = end_base - start_base
    t_opt = end_opt - start_opt
    
    improvement = (t_base - t_opt) / t_base
    
    results = {
        "steps": steps,
        "baseline_time": t_base,
        "optimized_time": t_opt,
        "throughput_improvement": improvement,
        "tokens_per_sec_opt": steps / t_opt,
        "intervention_reduction": 1.0 - controller.pulse_scheduler.get_pulse_frequency(),
        "pulse_count": controller.pulse_scheduler.get_telemetry()["pulse_count"]
    }
    
    print(f"Baseline Time: {t_base:.4f}s")
    print(f"Optimized Time: {t_opt:.4f}s")
    print(f"Throughput Improvement: {improvement:.2%}")
    print(f"Intervention Reduction: {results['intervention_reduction']:.2%}")
    
    # Save results
    os.makedirs("results/phase26", exist_ok=True)
    with open("results/phase26/throughput_recovery_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_throughput_benchmark()
