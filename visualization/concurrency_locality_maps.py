import matplotlib.pyplot as plt
import numpy as np
import os

def plot_concurrency_locality(zones: list, depths: list, save_path: str):
    """
    Visualizes request distribution across locality-aware zones.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(zones, depths, color='skyblue')
    plt.title("Locality-Aware Queue Balancing & Zone Affinity")
    plt.xlabel("Retrieval Zone")
    plt.ylabel("Pending Requests")
    plt.xticks(zones)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    zones = [0, 1, 2, 3]
    depths = [12, 15, 8, 20]
    plot_concurrency_locality(zones, depths, "results/phase7_5/concurrency_locality_map.png")
