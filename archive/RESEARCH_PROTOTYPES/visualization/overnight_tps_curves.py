import matplotlib.pyplot as plt
import os

class OvernightTPSCurves:
    """
    Generates detailed TPS curves for multi-hour runs.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5b"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_overnight_tps(self, hours: list, tps: list):
        plt.figure(figsize=(15, 7))
        plt.plot(hours, tps, label='Overnight TPS')
        plt.axhline(y=min(tps), color='r', linestyle='--', label='Min TPS')
        plt.xlabel('Time (Hours)')
        plt.ylabel('TPS')
        plt.title('Overnight TPS Stability')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'overnight_tps.png'))
        plt.close()

if __name__ == "__main__":
    v = OvernightTPSCurves()
    v.plot_overnight_tps(range(8), [50, 49, 48.5, 48, 47.8, 47.5, 47.2, 47])
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'overnight_tps.png')}")
