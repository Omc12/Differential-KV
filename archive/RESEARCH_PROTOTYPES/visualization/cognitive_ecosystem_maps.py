import matplotlib.pyplot as plt
import numpy as np

def plot_ecosystem_map(active_attractors: dict, save_path: str = "results/phase35/ecosystem_map.png"):
    """
    Plots a 2D projection of the attractor ecosystem.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for aid, meta in active_attractors.items():
        # Simulated 2D coordinates from 'center'
        center = np.random.randn(2) 
        radius = meta['density'] * 0.5
        
        color = 'green' if meta['health'] > 1.0 else 'orange'
        if meta.get('suppression'): color = 'red'
        
        circle = plt.Circle(center, radius, color=color, alpha=0.3)
        ax.add_patch(circle)
        ax.text(center[0], center[1], aid, fontsize=8)
        
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title("Cognitive Attractor Ecosystem Map")
    plt.savefig(save_path)
    print(f"Ecosystem map saved to {save_path}")
