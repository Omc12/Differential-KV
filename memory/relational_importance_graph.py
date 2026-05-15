import torch

class RelationalImportanceGraph:
    """
    PHASE 20.1B: Tracks relationships between tokens to propagate importance.
    If 'sk-ant' is important, 'api05' following it becomes relationally important.
    """
    def __init__(self, max_nodes: int = 2048):
        self.max_nodes = max_nodes
        self.edges = {} # (idx_a, idx_b) -> weight

    def add_relationship(self, idx_a: int, idx_b: int, weight: float = 1.0):
        """
        Adds a semantic edge between two tokens.
        """
        key = tuple(sorted((idx_a, idx_b)))
        self.edges[key] = self.edges.get(key, 0.0) + weight

    def propagate_importance(self, base_importance: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Adjusts importance weights based on relational connectivity.
        """
        flat_indices = indices.flatten().tolist()
        flat_imp = base_importance.flatten()
        
        # Simple graph-based propagation
        # If an index is important, its neighbors in the graph get a boost.
        new_imp = flat_imp.clone()
        for i, idx in enumerate(flat_indices):
            if flat_imp[i] > 0.5:
                # Find neighbors
                for (a, b), w in self.edges.items():
                    neighbor = b if a == idx else (a if b == idx else None)
                    if neighbor is not None and neighbor in flat_indices:
                        n_idx = flat_indices.index(neighbor)
                        new_imp[n_idx] += flat_imp[i] * w * 0.2
        
        return new_imp.view_as(base_importance).clamp(max=1.0)
