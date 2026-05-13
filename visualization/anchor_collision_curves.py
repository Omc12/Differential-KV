import matplotlib.pyplot as plt
import numpy as np
import os

def plot_anchor_collision_curves(steps: list, collisions: list, save_path: str):
    """
    Plots the collision frequency over time during adaptive optimization.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, collisions, marker='o', linestyle='-', color='orange')
    plt.title("Anchor Collision Frequency & Stabilization")
    plt.xlabel("Optimization Step")
    plt.ylabel("Collisions / Warp")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    steps = list(range(10))
    # Simulated reduction in collisions
    collisions = [25, 20, 18, 12, 8, 5, 4, 3, 3, 3]
    plot_anchor_collision_curves(steps, collisions, "results/phase7_5/anchor_collision_curves.png")
