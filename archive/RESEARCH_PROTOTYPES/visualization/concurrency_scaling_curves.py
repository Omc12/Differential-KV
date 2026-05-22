import matplotlib.pyplot as plt
import os

class ConcurrencyScalingCurves:
    """
    Generates throughput vs concurrent user curves.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5cde"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_scaling(self, users: list, tps: list):
        plt.figure(figsize=(10, 6))
        plt.plot(users, tps, marker='o')
        plt.xlabel('Concurrent Users')
        plt.ylabel('Total TPS')
        plt.title('Concurrency Scaling Curve')
        plt.savefig(os.path.join(self.output_dir, 'concurrency_scaling.png'))
        plt.close()

if __name__ == "__main__":
    v = ConcurrencyScalingCurves()
    v.plot_scaling([1, 2, 4, 8, 16], [150, 280, 520, 980, 1800])
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'concurrency_scaling.png')}")
