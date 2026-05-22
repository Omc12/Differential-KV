"""
visualization/memory_pressure_maps.py

Maps VRAM usage and migration events over time.
Focus: memory pressure maps, VRAM allocation patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

def plot_memory_pressure_map(vram_stats: List[Tuple[float, float]], save_path: str = "results/reconstruction_5a/memory_pressure_map.png"):
    steps = np.arange(len(vram_stats))
    allocated = [s[0] for s in vram_stats]
    reserved = [s[1] for s in vram_stats]
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(steps, reserved, label='Reserved VRAM', color='lightcoral', alpha=0.3)
    plt.plot(steps, allocated, label='Allocated VRAM', color='red', linewidth=2)
    
    plt.title("VRAM Pressure Over Time")
    plt.xlabel("Step")
    plt.ylabel("Memory (MB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Memory map saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Mock data: memory grows then stabilizes
    steps = 100
    mock_stats = []
    curr = 1024
    for i in range(steps):
        curr += np.random.randint(5, 15)
        mock_stats.append((curr, curr + 256))
        
    plot_memory_pressure_map(mock_stats)
