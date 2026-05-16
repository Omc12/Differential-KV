import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_manifold_evolution_timeline(drift_scores: List[float], stability_scores: List[float], save_path: str = "manifold_timeline.png"):
    """
    Plots the evolution of manifold drift and stability over time.
    """
    steps = np.arange(len(drift_scores))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Execution Horizon (Steps)')
    ax1.set_ylabel('Identity Drift', color='tab:red')
    ax1.plot(steps, drift_scores, color='tab:red', label='Drift')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reasoning Stability', color='tab:blue')
    ax2.plot(steps, stability_scores, color='tab:blue', label='Stability')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title("Persistent Manifold Evolution Timeline")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Demo data
    drift = np.cumsum(np.random.normal(0.001, 0.005, 100))
    stability = 0.95 + np.random.normal(0, 0.01, 100)
    plot_manifold_evolution_timeline(drift.tolist(), stability.tolist())
