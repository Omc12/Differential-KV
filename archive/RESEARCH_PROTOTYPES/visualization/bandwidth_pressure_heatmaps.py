import matplotlib.pyplot as plt
import numpy as np

def plot_bandwidth_heatmap(pressure_matrix, save_path):
    """
    PHASE 9: Bandwidth Pressure Heatmaps
    Heatmap of memory bandwidth utilization across the KV cache.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(pressure_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Bandwidth Usage (GB/s)')
    plt.xlabel("Heads")
    plt.ylabel("Context Blocks")
    plt.title("KV Bandwidth Pressure Heatmap")
    plt.savefig(save_path)
    plt.close()
