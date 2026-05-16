"""
visualization/latency_scaling_curves.py

Plots latency scaling across different concurrency levels and context lengths.
Focuses on P95 latency stability.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_latency_scaling(concurrency: np.ndarray, p95_latency: np.ndarray, avg_latency: np.ndarray):
    """
    Plots P95 and Average latency against concurrency.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(concurrency, p95_latency, marker='o', label='P95 Latency', color='red')
    plt.plot(concurrency, avg_latency, marker='s', label='Avg Latency', color='blue')
    plt.title('Latency Scaling vs. Concurrency')
    plt.xlabel('Concurrency (Concurrent Requests)')
    plt.ylabel('Latency (ms/token)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/plots/latency_scaling_curves.png')
    plt.close()

if __name__ == "__main__":
    # Mock data
    concurrency = np.array([1, 2, 4, 8, 16])
    p95 = np.array([25, 28, 35, 55, 110])
    avg = np.array([20, 22, 28, 40, 80])
    plot_latency_scaling(concurrency, p95, avg)
