import matplotlib.pyplot as plt
import pandas as pd
import os

class FrontierRuntimeComparison:
    """
    Generates a comparison dashboard between Differential KV 
    and frontier runtimes (vLLM, FA2, etc.).
    """
    def __init__(self, output_dir: str = "results/reconstruction_5cde"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_comparison(self, df: pd.DataFrame):
        """
        df columns: ['Runtime', 'TPS', 'VRAM_Usage', 'Retrieval_Stability']
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # TPS Comparison
        df.plot(x='Runtime', y='TPS', kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Throughput (TPS)')
        
        # VRAM Comparison
        df.plot(x='Runtime', y='VRAM_Usage', kind='bar', ax=axes[1], color='salmon')
        axes[1].set_title('VRAM Usage (MB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'frontier_comparison.png'))
        plt.close()

if __name__ == "__main__":
    v = FrontierRuntimeComparison()
    import pandas as pd
    df = pd.DataFrame({
        'Runtime': ['Differential KV', 'vLLM', 'FA2', 'KIVI'],
        'TPS': [180, 150, 120, 130],
        'VRAM_Usage': [2048, 6144, 4096, 1024]
    })
    v.plot_comparison(df)
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'frontier_comparison.png')}")
