"""
visualization/adaptive_energy_flow.py
Phase 27: Adaptive Cognitive Routing (ACR)
Visualizes pulse density and energy efficiency across regimes.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_energy_efficiency(results_path: str, save_path: str):
    with open(results_path, "r") as f:
        data = json.load(f)
        
    regimes = list(data.keys())
    pulse_densities = [data[r]["pulse_density"] for r in regimes]
    survivals = [data[r]["avg_survival"] for r in regimes]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(regimes))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    rects1 = ax1.bar(x - width/2, pulse_densities, width, label='Pulse Density', color='blue', alpha=0.7)
    ax1.set_ylabel('Pulse Density (%)')
    ax1.set_title('Resonance Efficiency by Cognitive Regime')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, rotation=45, ha='right')
    ax1.axhline(y=0.005, color='red', linestyle='--', label='0.5% Target')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, survivals, width, label='Reasoning Survival', color='green', alpha=0.7)
    ax2.set_ylabel('Survival Rate')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.75, color='orange', linestyle='--', label='75% Target')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Energy flow visualization saved to {save_path}")

if __name__ == "__main__":
    os.makedirs("results/phase27/plots", exist_ok=True)
    try:
        plot_energy_efficiency("results/phase27/regime_specific_results.json", "results/phase27/plots/energy_efficiency.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")
