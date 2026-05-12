"""
analysis/escape_theory_visualizations.py

Visualization suite for Phase 22 Escape Theory.
Generates basin maps, escape trajectories, and branch survival trees.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class EscapeTheoryVisualizer:
    def __init__(self, output_dir: str = "results/phase22/plots/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="darkgrid")

    def plot_collapse_basin_map(self, basin_data: Dict[str, np.ndarray]):
        """
        Generates a 2D/3D map of the latent manifold showing collapse basins.
        """
        plt.figure(figsize=(10, 8))
        # Simulated basin map (heatmap of health scores over latent dimensions)
        data = basin_data.get("manifold", np.random.rand(50, 50))
        sns.heatmap(data, cmap="RdYlGn_r", cbar_kws={'label': 'Collapse Probability'})
        plt.title("Latent Manifold: Collapse Basin Map")
        plt.xlabel("Latent Principal Dimension 1")
        plt.ylabel("Latent Principal Dimension 2")
        plt.savefig(os.path.join(self.output_dir, "collapse_basin_map.png"))
        plt.close()

    def plot_escape_trajectories(self, trajectories: List[Dict[str, Any]]):
        """
        Plots multiple trajectories trying to escape a basin.
        """
        plt.figure(figsize=(10, 6))
        for i, traj in enumerate(trajectories):
            steps = np.arange(len(traj["health"]))
            plt.plot(steps, traj["health"], label=traj.get("name", f"Trajectory {i}"), 
                     alpha=0.8, linewidth=2 if traj.get("is_best", False) else 1)
            
            # Mark interventions
            for intv_step in traj.get("interventions", []):
                plt.scatter(intv_step, traj["health"][intv_step], marker='x', color='red')

        plt.axhline(y=0.3, color='black', linestyle='--', label="Irreversibility Threshold")
        plt.title("Recovery Dynamics: Escape Trajectories")
        plt.xlabel("Inference Step")
        plt.ylabel("Cognitive Health Score")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "escape_trajectories.png"))
        plt.close()

    def plot_branch_survival_tree(self, tree_data: Dict[str, Any]):
        """
        Visualizes the branching reasoning paths.
        """
        plt.figure(figsize=(12, 8))
        # Schematic representation of branching
        # In a real implementation, we'd use networkx
        plt.text(0.5, 0.9, "Primary Trajectory", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
        
        # Draw branches
        for i in range(3):
            plt.arrow(0.5, 0.85, (i-1)*0.2, -0.3, head_width=0.02)
            plt.text(0.5 + (i-1)*0.2, 0.5, f"Branch {i}\nSurvival: {0.3 + i*0.2:.1f}", ha='center')
            
        plt.title("Reasoning Manifold Branching Tree")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, "branch_survival_tree.png"))
        plt.close()

    def plot_rollback_success_heatmap(self, rollback_data: np.ndarray):
        """
        Heatmap of recovery probability vs Rollback Distance and Initial Health.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(rollback_data, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Rollback Success Probability")
        plt.xlabel("Rewind Distance (Steps)")
        plt.ylabel("Collapse Depth at Detection")
        plt.savefig(os.path.join(self.output_dir, "rollback_success_heatmap.png"))
        plt.close()

    def plot_saturation_curves(self, curves: Dict[str, List[float]]):
        """
        Plots repair saturation (Intervention Count vs Health).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(curves["intervention_counts"], curves["health_scores"], marker='o', alpha=0.6)
        plt.title("Repair Saturation Analysis")
        plt.xlabel("Cumulative Intervention Count")
        plt.ylabel("Cognitive Health Score")
        plt.savefig(os.path.join(self.output_dir, "repair_saturation.png"))
        plt.close()
