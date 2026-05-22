import matplotlib.pyplot as plt
import os

class DriftCurves:
    """
    Generates plots for performance drift over time.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5b"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_tps_drift(self, steps: list, tps: list):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, tps, label='TPS')
        plt.xlabel('Step')
        plt.ylabel('Tokens Per Second')
        plt.title('Long-Horizon TPS Drift')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'tps_drift.png'))
        plt.close()

if __name__ == "__main__":
    # Test plotting
    v = DriftCurves()
    v.plot_tps_drift(range(100), [50 - (i*0.1) for i in range(100)])
