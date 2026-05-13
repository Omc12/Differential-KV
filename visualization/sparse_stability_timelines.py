import matplotlib.pyplot as plt
import os

class SparseStabilityTimelines:
    """
    Generates timelines for sparse execution stability and oscillation.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5b"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_stability(self, steps: list, density: list, oscillations: list):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Density', color='tab:blue')
        ax1.plot(steps, density, color='tab:blue', label='Density')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Oscillations', color='tab:orange')
        ax2.bar(steps, oscillations, color='tab:orange', alpha=0.3, label='Oscillations')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title('Sparse Stability Timeline')
        plt.savefig(os.path.join(self.output_dir, 'sparse_stability.png'))
        plt.close()

if __name__ == "__main__":
    v = SparseStabilityTimelines()
    v.plot_stability(range(100), [0.15 + (i*0.0001) for i in range(100)], [int(i%10==0) for i in range(100)])
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'sparse_stability.png')}")
