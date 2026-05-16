"""
visualization/collective_cognition_maps.py

Generates collective reasoning graphs and maps of distributed cognition.
"""

import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any

def generate_collective_cognition_map(agents: Dict[str, Any], output_path: str = "collective_map.png"):
    """
    Plots a map of agents and their shared manifolds.
    """
    plt.figure(figsize=(10, 8))
    
    # Simple scatter plot of agents in a 2D resonance space
    for i, (aid, meta) in enumerate(agents.items()):
        x = hash(aid) % 100
        y = i * 20
        plt.scatter(x, y, s=500, alpha=0.6, label=aid)
        plt.text(x, y, aid, fontsize=12, ha='center')
        
    plt.title("Collective Cognition Map")
    plt.xlabel("Cognitive Resonance")
    plt.ylabel("Agent Hierarchy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()
    print(f"Generated collective cognition map at {output_path}")

if __name__ == "__main__":
    mock_agents = {
        "Agent_A": {"resonance": 0.95},
        "Agent_B": {"resonance": 0.88},
        "Agent_C": {"resonance": 0.92}
    }
    generate_collective_cognition_map(mock_agents)
