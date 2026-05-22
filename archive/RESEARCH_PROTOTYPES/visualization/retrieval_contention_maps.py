import matplotlib.pyplot as plt
import os

class RetrievalContentionMaps:
    """
    Visualizes VRAM contention across concurrent retrieval sessions.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5cde"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_contention(self, block_ids: list, collision_counts: list):
        plt.figure(figsize=(12, 6))
        plt.bar(block_ids, collision_counts, color='orange')
        plt.xlabel('VRAM Block ID')
        plt.ylabel('Collision Count')
        plt.title('Retrieval Contention Map')
        plt.savefig(os.path.join(self.output_dir, 'retrieval_contention.png'))
        plt.close()
