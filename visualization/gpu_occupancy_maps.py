import matplotlib.pyplot as plt
import numpy as np

def plot_gpu_occupancy(timestamps, occupancy_rates, save_path):
    """
    PHASE 9: GPU Occupancy Maps
    Visualizes SM utilization over the course of an inference pass.
    """
    plt.figure(figsize=(12, 4))
    plt.fill_between(timestamps, occupancy_rates, alpha=0.3)
    plt.plot(timestamps, occupancy_rates, color='green')
    plt.ylim(0, 1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Occupancy")
    plt.title("GPU SM Utilization Timeline")
    plt.savefig(save_path)
    plt.close()
