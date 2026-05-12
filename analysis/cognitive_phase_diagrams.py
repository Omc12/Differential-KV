"""
analysis/cognitive_phase_diagrams.py
Phase 19: Universal Cognitive Geometry
Builds phase diagrams for compressed cognition (e.g., Compression vs Coherence).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

class CognitivePhaseDiagram:
    def __init__(self, x_label="Compression", y_label="Coherence"):
        self.x_data = []
        self.y_data = []
        self.z_data = [] # Stability/Recovery
        self.x_label = x_label
        self.y_label = y_label

    def add_point(self, x: float, y: float, z: float):
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

    def plot_diagram(self, save_path: str):
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(self.x_data, self.y_data, c=self.z_data, cmap='viridis', s=100, edgecolors='white')
        plt.colorbar(sc, label="Stability Score")
        
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(f"Cognitive Phase Diagram: {self.x_label} vs {self.y_label}")
        plt.grid(True, alpha=0.2)
        
        # Add phase boundary estimate if enough data
        if len(self.x_data) > 10:
            # Simple threshold boundary for illustration
            pass

        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    cpd = CognitivePhaseDiagram()
    # Mock data
    for comp in np.linspace(0.1, 0.9, 10):
        for coh in np.linspace(0.1, 0.9, 10):
            stability = max(0, 1.0 - (comp * 0.8 + (1-coh) * 0.5))
            cpd.add_point(comp, coh, stability)
            
    import os
    os.makedirs("results/phase19/plots", exist_ok=True)
    cpd.plot_diagram("results/phase19/plots/phase_diagram_mock.png")
