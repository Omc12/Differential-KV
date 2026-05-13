import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gpu_runtime_map(kernel_times: dict, save_path: str):
    """
    Plots a pie chart or bar chart of where GPU time is spent 
    in the sparse runtime.
    """
    labels = list(kernel_times.keys())
    sizes = list(kernel_times.values())
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title("GPU Sparse Runtime Latency Distribution")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    times = {"Retrieval": 45, "Reconstruction": 30, "Attention": 20, "Overhead": 5}
    plot_gpu_runtime_map(times, "results/phase7/gpu_runtime_map.png")
