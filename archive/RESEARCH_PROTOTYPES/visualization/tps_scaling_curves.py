import matplotlib.pyplot as plt
import numpy as np

def plot_tps_scaling(context_lengths, tps_values, labels, save_path):
    """
    PHASE 9: TPS Scaling Curves
    Plots E2E TPS as a function of context length (32k to 1M).
    Demonstrates the efficiency of sparse acceleration.
    """
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(context_lengths, tps_values[i], marker='o', label=label)
        
    plt.xlabel("Context Length")
    plt.ylabel("Tokens Per Second (TPS)")
    plt.title("E2E Throughput Scaling (Differential KV Phase 6)")
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
