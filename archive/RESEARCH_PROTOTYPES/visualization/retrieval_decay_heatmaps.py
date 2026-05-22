import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class RetrievalDecayHeatmaps:
    """
    Generates heatmaps for retrieval survival across different contexts.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5b"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_heatmap(self, decay_matrix: np.ndarray, labels: list):
        plt.figure(figsize=(12, 8))
        sns.heatmap(decay_matrix, xticklabels=labels, yticklabels=labels, cmap="YlOrRd")
        plt.title('Retrieval Decay Heatmap')
        plt.savefig(os.path.join(self.output_dir, 'retrieval_decay_heatmap.png'))
        plt.close()

if __name__ == "__main__":
    v = RetrievalDecayHeatmaps()
    data = np.random.rand(10, 10)
    v.plot_heatmap(data, [f"C{i}" for i in range(10)])
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'retrieval_decay_heatmap.png')}")
