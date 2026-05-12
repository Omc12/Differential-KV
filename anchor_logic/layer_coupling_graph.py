import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

class LayerCouplingGraph:
    """
    Represents the inter-layer dependencies and resonance paths.
    Used to determine which layers can stabilize each other.
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_layers))
        # Initial edges: sequential coupling
        for i in range(num_layers - 1):
            self.graph.add_edge(i, i+1, weight=1.0)
            
    def update_coupling(self, alignment_matrix: np.ndarray):
        """Updates edge weights based on measured resonance alignment."""
        for i in range(self.num_layers):
            for j in range(i + 1, self.num_layers):
                alignment = alignment_matrix[i, j]
                if alignment > 0.3: # Threshold for resonance coupling
                    self.graph.add_edge(i, j, weight=float(alignment))
                elif self.graph.has_edge(i, j) and abs(i - j) > 1:
                    # Remove weak long-range couplings
                    self.graph.remove_edge(i, j)
                    
    def get_stabilizing_neighbors(self, layer_idx: int) -> List[Tuple[int, float]]:
        """
        Returns a list of (neighbor_idx, weight) for layers that can 
        help stabilize the target layer.
        """
        neighbors = []
        if layer_idx in self.graph:
            for neighbor in self.graph.neighbors(layer_idx):
                weight = self.graph[layer_idx][neighbor]['weight']
                neighbors.append((neighbor, weight))
        
        # Sort by weight descending
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def get_global_sync_state(self) -> float:
        """Returns the algebraic connectivity of the graph as a proxy for global sync."""
        if not nx.is_connected(self.graph):
            return 0.0
        # For a more robust metric, we could use the second smallest eigenvalue of the Laplacian
        # but for speed, we'll use average clustering coefficient
        return nx.average_clustering(self.graph, weight='weight')
