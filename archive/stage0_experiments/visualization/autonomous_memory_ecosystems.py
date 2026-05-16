import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List

def plot_memory_ecosystem_graph(categories: Dict[str, List[str]], save_path: str = "memory_ecosystem.png"):
    """
    Visualizes the hierarchical memory ecosystem and motif relationships.
    """
    G = nx.Graph()
    G.add_node("Global Substrate")
    
    for category, motifs in categories.items():
        G.add_node(category)
        G.add_edge("Global Substrate", category)
        for motif in motifs:
            G.add_node(motif)
            G.add_edge(category, motif)
            
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10)
    plt.title("Autonomous Memory Ecosystem Map")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Demo data
    data = {
        "Mathematics": ["Calculus Motif", "Topology Motif", "Algebra Motif"],
        "Coding": ["Python Motif", "Rust Motif", "Kernel Motif"],
        "Planning": ["Recursive Goal Motif", "Heuristic Motif"]
    }
    plot_memory_ecosystem_graph(data)
