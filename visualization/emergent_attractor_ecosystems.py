"""
visualization/emergent_attractor_ecosystems.py

Visualizes the ecosystem of emergent collaborative attractors.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def plot_attractor_ecosystem(attractors: List[Dict[str, Any]], output_path: str = "attractor_ecosystem.png"):
    """
    Plots the distribution and stability of emergent attractors.
    """
    plt.figure(figsize=(12, 6))
    
    names = [a['id'] for a in attractors]
    stability = [a['stability'] for a in attractors]
    reuse = [a['reuse'] for a in attractors]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, stability, width, label='Stability')
    plt.bar(x + width/2, reuse, width, label='Reuse Rate')
    
    plt.xticks(x, names, rotation=45)
    plt.ylabel('Score')
    plt.title('Emergent Attractor Ecosystem')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Generated attractor ecosystem visualization at {output_path}")

if __name__ == "__main__":
    mock_attractors = [
        {'id': 'logic_A', 'stability': 0.98, 'reuse': 0.85},
        {'id': 'code_B', 'stability': 0.96, 'reuse': 0.78},
        {'id': 'plan_C', 'stability': 0.99, 'reuse': 0.92},
    ]
    plot_attractor_ecosystem(mock_attractors)
