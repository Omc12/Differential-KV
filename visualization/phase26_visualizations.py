"""
visualization/phase26_visualizations.py
Phase 26: Cognitive Energy Minimization (CEM)
Generates energy landscapes, basin maps, and efficiency charts.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_energy_landscape(data_path="results/phase26/energy_curves.json"):
    """Visualizes the cognitive energy trajectory and basin boundaries."""
    if not os.path.exists(data_path): 
        print(f"Data not found: {data_path}")
        return
        
    with open(data_path, "r") as f:
        data = json.load(f)
    
    energy_curve = data["energy_curve"]
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    # Gradient fill for basins
    plt.fill_between(range(len(energy_curve)), 0, 0.15, color='green', alpha=0.1, label='Stable Basin')
    plt.fill_between(range(len(energy_curve)), 0.15, 0.4, color='yellow', alpha=0.1, label='Semi-Stable')
    plt.fill_between(range(len(energy_curve)), 0.4, max(energy_curve)+0.1, color='red', alpha=0.1, label='Collapse Basin')
    
    plt.plot(energy_curve, color='#00ffcc', linewidth=1.5, label='Cognitive Energy (E)')
    
    plt.title("Cognitive Energy Landscape (Phase 26)", fontsize=14, pad=15)
    plt.xlabel("Reasoning Steps", fontsize=12)
    plt.ylabel("Energy Intensity", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("results/phase26/energy_landscape.png")
    plt.close()
    print("Generated energy_landscape.png")

def plot_stability_and_pulses(data_path="results/phase26/minimal_intervention_results.json"):
    """Visualizes how sparse pulses recover stability."""
    if not os.path.exists(data_path): 
        print(f"Data not found: {data_path}")
        return
        
    with open(data_path, "r") as f:
        data = json.load(f)
    
    stability = data["stability_trend"]
    pulses = data["pulse_steps"]
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    plt.plot(stability, color='#ff00ff', label='Reasoning Stability', linewidth=2)
    
    # Mark pulses
    first_pulse = True
    for p in pulses:
        plt.axvline(x=p, color='orange', linestyle=':', alpha=0.6, 
                    label='Resonance Pulse' if first_pulse else "")
        plt.scatter(p, stability[p], color='orange', zorder=5)
        first_pulse = False
        
    plt.title("Stability Preservation via Sparse Reinforcement (CEM)", fontsize=14, pad=15)
    plt.xlabel("Reasoning Steps", fontsize=12)
    plt.ylabel("Stability Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("results/phase26/stability_pulses.png")
    plt.close()
    print("Generated stability_pulses.png")

def plot_basin_transitions(data_path="results/phase26/energy_curves.json"):
    """Generates a bar chart of basin distribution."""
    if not os.path.exists(data_path): return
    with open(data_path, "r") as f:
        data = json.load(f)
        
    dist = data["basin_distribution"]
    basins = list(dist.keys())
    counts = list(dist.values())
    
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 6))
    colors = ['#00ff00', '#ffff00', '#ff0000']
    plt.bar(basins, counts, color=colors, alpha=0.7)
    
    plt.title("Reasoning Basin Distribution", fontsize=14)
    plt.ylabel("Step Count", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("results/phase26/basin_distribution.png")
    plt.close()
    print("Generated basin_distribution.png")

if __name__ == "__main__":
    plot_energy_landscape()
    plot_stability_and_pulses()
    plot_basin_transitions()
