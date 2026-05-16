"""
visualization/regime_transition_maps.py
Phase 27: Adaptive Cognitive Routing (ACR)
Visualizes regime transitions and detection accuracy.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_regime_transitions(results_path: str, save_path: str):
    with open(results_path, "r") as f:
        data = json.load(f)
        
    history = data["history_sample"]
    steps = [h["step"] for h in history]
    target_regimes = [h["target_regime"] for h in history]
    detected_regimes = [h["detected_regime"] for h in history]
    overhead = [h["overhead"] for h in history]
    
    # Map regimes to numbers for plotting
    unique_regimes = sorted(list(set(target_regimes + detected_regimes)))
    regime_map = {r: i for i, r in enumerate(unique_regimes)}
    
    target_nums = [regime_map[r] for r in target_regimes]
    detected_nums = [regime_map[r] for r in detected_regimes]
    
    plt.figure(figsize=(15, 7))
    
    # Subplot 1: Regime Transitions
    plt.subplot(2, 1, 1)
    plt.step(steps, target_nums, where='post', label="Target Regime", linestyle="--", alpha=0.5, color='gray')
    plt.scatter(steps, detected_nums, label="Detected Regime", marker='o', alpha=0.8, color='blue')
    plt.yticks(range(len(unique_regimes)), unique_regimes)
    plt.ylabel("Regime")
    plt.title("Cognitive Regime Transition Map")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Resource Overhead
    plt.subplot(2, 1, 2)
    plt.plot(steps, overhead, label="Geometry Overhead", color='green', marker='s')
    plt.axhline(y=0.06, color='red', linestyle='--', label="6% Limit")
    plt.ylabel("Overhead (%)")
    plt.xlabel("Step")
    plt.title("Adaptive Geometry Allocation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Transition map saved to {save_path}")

if __name__ == "__main__":
    os.makedirs("results/phase27/plots", exist_ok=True)
    try:
        plot_regime_transitions("results/phase27/mixed_cognition_results.json", "results/phase27/plots/regime_transitions.png")
    except Exception as e:
        print(f"Could not generate plot (maybe run evaluation first?): {e}")
