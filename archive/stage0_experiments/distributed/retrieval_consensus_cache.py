import torch
from typing import Dict, List

class RetrievalConsensusCache:
    """
    Global KV consensus: nodes share 'importance votes' for tokens.
    Ensures that the most relevant anchors are cached globally across the cluster.
    """
    def __init__(self, cluster_size: int):
        self.cluster_size = cluster_size
        self.global_votes: Dict[int, torch.Tensor] = {}

    def submit_votes(self, layer_id: int, importance_scores: torch.Tensor):
        """
        Nodes submit their local importance scores for a layer.
        importance_scores: [K_LEN]
        """
        if layer_id not in self.global_votes:
            self.global_votes[layer_id] = importance_scores
        else:
            # Average votes across the cluster
            self.global_votes[layer_id] = (self.global_votes[layer_id] + importance_scores) / 2

    def get_consensus_anchors(self, layer_id: int, top_k: int = 128) -> torch.Tensor:
        """Returns the indices of the globally most important tokens."""
        if layer_id not in self.global_votes:
            return torch.tensor([], dtype=torch.long)
        _, indices = torch.topk(self.global_votes[layer_id], top_k)
        return indices
