import matplotlib.pyplot as plt
import os

class FragmentationGrowthPlotter:
    """
    Generates plots for memory fragmentation growth.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5b"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_fragmentation(self, steps: list, frag_ratios: list):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, frag_ratios, color='red', label='Fragmentation')
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.title('Memory Fragmentation Growth')
        plt.savefig(os.path.join(self.output_dir, 'fragmentation_growth.png'))
        plt.close()

if __name__ == "__main__":
    v = FragmentationGrowthPlotter()
    v.plot_fragmentation(range(100), [0.05 + (i*0.002) for i in range(100)])
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'fragmentation_growth.png')}")
