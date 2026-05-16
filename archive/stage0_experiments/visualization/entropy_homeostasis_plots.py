import matplotlib.pyplot as plt

def plot_entropy_homeostasis(entropy_traj, target_entropy, save_path="results/phase35/entropy_homeostasis.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(entropy_traj, label="Current Entropy", color='blue')
    plt.axhline(y=target_entropy, color='red', linestyle='--', label="Equilibrium Target")
    plt.fill_between(range(len(entropy_traj)), target_entropy - 0.1, target_entropy + 0.1, color='green', alpha=0.1, label="Homeostasis Zone")
    plt.title("Entropy Homeostasis Tracking")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Shannon Entropy")
    plt.legend()
    plt.savefig(save_path)
    print(f"Homeostasis plot saved to {save_path}")
