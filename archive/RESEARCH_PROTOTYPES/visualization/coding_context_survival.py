import matplotlib.pyplot as plt
import os

class CodingContextSurvival:
    """
    Plots the survival rate of code anchors during long-horizon refactor sessions.
    """
    def __init__(self, output_dir: str = "results/reconstruction_5cde"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_survival(self, steps: list, survival_rates: list):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, survival_rates, color='green')
        plt.axhline(y=0.9, color='r', linestyle='--', label='Critical Threshold')
        plt.xlabel('Refactor Step')
        plt.ylabel('Anchor Survival Rate')
        plt.title('Coding Context Survival')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'coding_context_survival.png'))
        plt.close()

if __name__ == "__main__":
    v = CodingContextSurvival()
    v.plot_survival(range(10), [1.0, 0.99, 0.98, 0.98, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91])
    print(f"Generated sample graph: {os.path.join(v.output_dir, 'coding_context_survival.png')}")
