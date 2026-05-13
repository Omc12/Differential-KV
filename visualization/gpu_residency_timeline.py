"""
visualization/gpu_residency_timeline.py

Visualizes VRAM residency of anchors and sparse components over time.
Tracks memory allocation and eviction events.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_residency_timeline(timeline_data: dict):
    """
    Plots a Gantt-like chart of memory residency for different components.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    components = list(timeline_data.keys())
    for i, comp in enumerate(components):
        intervals = timeline_data[comp]
        for start, end in intervals:
            ax.barh(i, end - start, left=start, color=plt.cm.tab10(i % 10), alpha=0.8)
            
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components)
    ax.set_xlabel('Time (s)')
    ax.set_title('GPU Residency Timeline')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/gpu_residency_timeline.png')
    plt.close()

if __name__ == "__main__":
    # Mock data
    mock_data = {
        "Anchor_Set_A": [(0, 100), (150, 300)],
        "Sparse_Buffer_1": [(10, 80), (120, 250)],
        "L1_Cache": [(0, 300)],
        "Anchor_Set_B": [(50, 120), (200, 280)]
    }
    plot_residency_timeline(mock_data)
