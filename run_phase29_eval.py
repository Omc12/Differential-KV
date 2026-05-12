"""
run_phase29_eval.py

Main execution script for Phase 29: Kernel-Level Cognitive Runtime Acceleration.
Runs all benchmarks, scaling studies, and visualizations.
"""

import os
import subprocess
import time

def run_step(name, script_path):
    print(f"\n>>> Running {name}...")
    start = time.time()
    try:
        # Run with current python environment
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'
        result = subprocess.run(["py", script_path], capture_output=True, text=True, env=env)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error in {name}:")
            print(result.stderr)
    except Exception as e:
        print(f"Failed to execute {name}: {e}")
    print(f"<<< {name} completed in {time.time() - start:.2f}s")

def main():
    os.makedirs("results/phase29/viz", exist_ok=True)
    
    # 1. Benchmarks
    run_step("Kernel Fusion Benchmarks", "runtime/kernel_fusion_benchmarks.py")
    run_step("GPU Scaling Evaluation", "experiments/gpu_scaling_eval.py")
    run_step("Batched Agent Evaluation", "experiments/batched_agent_eval.py")
    
    # 2. Visualizations
    run_step("Execution Flow Viz", "visualization/kernel_execution_flow.py")
    run_step("Resonance Heatmaps", "visualization/gpu_resonance_heatmaps.py")
    run_step("Bandwidth Efficiency", "visualization/memory_bandwidth_maps.py")
    
    print("\nPhase 29 Evaluation Complete. Results stored in results/phase29/")

if __name__ == "__main__":
    main()
