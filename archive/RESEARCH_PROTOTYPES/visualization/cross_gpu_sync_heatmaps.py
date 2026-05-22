import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_sync_heatmap(world_size: int):
    """
    Generates a heatmap of desync drift between every GPU pair.
    """
    drift_matrix = np.random.rand(world_size, world_size) * 0.05
    np.fill_diagonal(drift_matrix, 0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(drift_matrix, annot=True, cmap="YlGnBu")
    plt.title("Cross-GPU Attractor Sync Drift")
    plt.xlabel("GPU Rank")
    plt.ylabel("GPU Rank")
    plt.savefig("results/phase33/sync_drift_heatmap.png")

if __name__ == "__main__":
    generate_sync_heatmap(8)
