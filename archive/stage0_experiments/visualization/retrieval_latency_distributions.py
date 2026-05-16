import matplotlib.pyplot as plt
import numpy as np

def plot_retrieval_distribution(latencies, save_path):
    """
    PHASE 9: Retrieval Latency Distributions
    Distribution of latencies for KV retrieval across the context.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=50, alpha=0.7, color='purple')
    plt.axvline(np.mean(latencies), color='r', linestyle='dashed', linewidth=1, label='Mean')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.title("KV Retrieval Latency Distribution")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
