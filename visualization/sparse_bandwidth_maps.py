"""
visualization/sparse_bandwidth_maps.py

Visualizes memory bandwidth utilization and sparse traffic locality.
Maps VRAM access patterns across different memory banks.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_bandwidth_map(traffic_data: np.ndarray, title: str = "Sparse Bandwidth Distribution"):
    """
    Plots a map of bandwidth consumption across GPU memory channels.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(traffic_data)), traffic_data, color='skyblue')
    plt.title(title)
    plt.xlabel('Memory Channel / Bank')
    plt.ylabel('Throughput (GB/s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/plots/sparse_bandwidth_map.png')
    plt.close()

if __name__ == "__main__":
    # Mock data: 32 memory channels
    mock_traffic = np.random.uniform(200, 800, 32)
    plot_bandwidth_map(mock_traffic)
