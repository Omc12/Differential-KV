import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

class QueueLatencyHeatmaps:
    """
    Generates heatmaps of queue latency under different load conditions.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5cde"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_heatmap(self, data: np.ndarray):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap="Reds")
        plt.title('Queue Latency Heatmap (Load vs Priority)')
        plt.savefig(os.path.join(self.output_dir, 'queue_latency_heatmap.png'))
        plt.close()
