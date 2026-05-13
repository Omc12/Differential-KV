import matplotlib.pyplot as plt
import os

def plot_concurrency_recovery(times: list, latencies: list, concurrency_limits: list, save_path: str):
    """
    Plots the relationship between system load, latency, and the 
    adaptive concurrency limit.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('P95 Latency (ms)', color=color)
    ax1.plot(times, latencies, color=color, label="P95 Latency")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Concurrency Limit', color=color)
    ax2.step(times, concurrency_limits, color=color, label="Concurrency Limit", where='post')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Concurrency Stabilization & Latency Recovery Curves")
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    times = list(range(60))
    latencies = [20 + (i**2 if i < 10 else 100 - i) for i in range(60)]
    limits = [8 if l < 50 else 4 for l in latencies]
    plot_concurrency_recovery(times, latencies, limits, "results/phase7/concurrency_recovery_curves.png")
