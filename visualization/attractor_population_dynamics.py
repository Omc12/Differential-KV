import matplotlib.pyplot as plt

def plot_population_dynamics(pop_history, save_path="results/phase35/population_dynamics.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(pop_history, label="Attractor Population", color='purple')
    plt.title("Attractor Ecology: Population Dynamics")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Active Attractor Count")
    plt.legend()
    plt.savefig(save_path)
    print(f"Population dynamics plot saved to {save_path}")
