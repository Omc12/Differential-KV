import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class RecursiveAttractorDynamics:
    """
    Analyzes the stability and persistence of recursive attractors.
    """
    def __init__(self):
        self.coherence_history = []
        self.reinforcement_gains = []
        self.attractor_drift = []

    def log_step(self, coherence: float, gain: float, drift: float):
        self.coherence_history.append(coherence)
        self.reinforcement_gains.append(gain)
        self.attractor_drift.append(drift)

    def calculate_persistence(self) -> float:
        if not self.coherence_history:
            return 0.0
        return np.mean(self.coherence_history)

    def plot_dynamics(self, save_path: str):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.coherence_history, label='Resonance Coherence')
        plt.axhline(y=0.7, color='r', linestyle='--', label='Collapse Threshold')
        plt.title('Recursive Attractor Coherence')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(self.reinforcement_gains, color='green', label='Reinforcement Gain')
        plt.title('Reinforcement Schedule')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(self.attractor_drift, color='orange', label='Latent Drift')
        plt.title('Attractor Manifold Drift')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def get_summary(self) -> Dict[str, float]:
        return {
            "mean_coherence": np.mean(self.coherence_history) if self.coherence_history else 0,
            "max_drift": np.max(self.attractor_drift) if self.attractor_drift else 0,
            "total_reinforcements": len([g for g in self.reinforcement_gains if g > 0])
        }
