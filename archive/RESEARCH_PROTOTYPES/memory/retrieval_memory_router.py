import torch
from typing import List, Dict, Optional
from .hierarchical_summary_memory import HierarchicalSummaryMemory

class RetrievalMemoryRouter:
    """
    Routes queries between active context and hierarchical summary memory.
    Ensures bounded context windows while maintaining retrieval continuity.
    """
    def __init__(self, memory: HierarchicalSummaryMemory, top_k: int = 5):
        self.memory = memory
        self.top_k = top_k
        
    def route_query(self, query: str, active_context_tokens: List[str]) -> str:
        """
        Determines the optimal context bridge for a given query.
        """
        # In a production system, this would use an embedding-based search
        # over the summary library.
        # For this implementation, we use the hierarchical bridge from memory.
        
        # 1. Check if the query can be satisfied by the active context.
        # (This is usually handled by the transformer itself, but the router 
        # decides what EXTRA information to inject).
        
        # 2. Get relevant summaries from the hierarchical memory.
        context_bridge = self.memory.get_context_bridge(query)
        
        # 3. Construct the augmented context.
        # Note: The actual injection happens at the prompt level.
        return context_bridge

    def get_sparse_retrieval_indices(self, query_vector: torch.Tensor, key_vectors: torch.Tensor) -> torch.Tensor:
        """
        Hardware-aware retrieval: returns indices of the most relevant KV pairs.
        Used for sparse attention routing.
        """
        # query_vector: [D]
        # key_vectors: [N, D]
        scores = torch.matmul(key_vectors, query_vector)
        _, indices = torch.topk(scores, min(self.top_k, scores.size(0)))
        return indices
