import torch

class SemanticHeadActivator:
    """
    Selectively activates attention heads based on semantic retrieval relevance.
    """
    def __init__(self, num_heads: int):
        self.num_heads = num_heads
        self.activation_history = []

    def get_active_heads(self, retrieval_relevance: torch.Tensor) -> torch.Tensor:
        # retrieval_relevance: [heads]
        # Keep only heads with relevance > median
        threshold = torch.median(retrieval_relevance)
        mask = retrieval_relevance >= threshold
        self.activation_history.append(mask.float().mean().item())
        return mask

    def get_activation_ratio(self):
        if not self.activation_history:
            return 1.0
        return sum(self.activation_history) / len(self.activation_history)
