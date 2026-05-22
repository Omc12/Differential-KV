import matplotlib.pyplot as plt
import numpy as np

def plot_anchor_collision_curves(static_collisions, adaptive_collisions, save_path="reports/anchor_collisions.png"):
    """
    Plots the collision frequency over time for static vs adaptive anchoring.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(static_collisions, label="Static Anchors", color="red", linestyle="--")
    plt.plot(adaptive_collisions, label="Adaptive Anchors (7.5A)", color="green")
    
    plt.title("Anchor Collision Frequency: Static vs. Adaptive")
    plt.xlabel("Time Steps")
    plt.ylabel("Collision Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    print(f"Saved collision curves to {save_path}")

if __name__ == "__main__":
    # Sample data for testing
    steps = 100
    static = 0.1 + 0.05 * np.random.randn(steps)
    adaptive = 0.02 + 0.01 * np.random.randn(steps)
    plot_anchor_collision_curves(static, adaptive)
