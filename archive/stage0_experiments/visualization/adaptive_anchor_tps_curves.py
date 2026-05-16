import matplotlib.pyplot as plt
import numpy as np

def plot_adaptive_anchor_tps_curves(tps_history, anchor_counts, save_path="reports/tps_anchor_dynamics.png"):
    """
    Plots TPS vs. Anchor Count to show the impact of adaptive density on performance.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('TPS (Tokens Per Second)', color=color)
    ax1.plot(tps_history, color=color, label="TPS")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Active Anchor Count', color=color)
    ax2.plot(anchor_counts, color=color, alpha=0.5, label="Anchor Count")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("TPS Stability vs. Adaptive Anchor Scaling")
    fig.tight_layout()
    plt.grid(True, alpha=0.2)
    
    plt.savefig(save_path)
    print(f"Saved TPS/Anchor dynamics to {save_path}")

if __name__ == "__main__":
    steps = 200
    tps = 150 + 10 * np.sin(np.linspace(0, 10, steps)) + 5 * np.random.randn(steps)
    anchors = 1024 + 512 * np.cos(np.linspace(0, 10, steps))
    plot_adaptive_anchor_tps_curves(tps, anchors)
