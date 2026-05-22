import torch

class TraversalRoutePreserver:
    """
    PHASE 19.0D: Traversal Route Preserver.
    Tracks successful traversal routes (sequences of high attention) 
    and pins them to prevent future pruning.
    """
    def __init__(self, persistence_limit: int = 100):
        self.routes = [] # List of (query_idx, target_indices)
        self.persistence_limit = persistence_limit

    def record_traversal(self, query_idx: int, target_indices: torch.Tensor):
        self.routes.append((query_idx, target_indices))
        if len(self.routes) > self.persistence_limit:
            self.routes.pop(0)

    def get_pinned_indices(self) -> torch.Tensor:
        if not self.routes:
            return torch.tensor([], dtype=torch.long)
        
        all_indices = []
        for _, targets in self.routes:
            all_indices.append(targets.cpu())
            
        return torch.unique(torch.cat(all_indices))
