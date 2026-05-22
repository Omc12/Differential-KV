import torch

class AdaptiveAnchorDensity:
    """
    Dynamically adjusts the number of protected anchor tokens based on 
    context entropy and retrieval failure rates.
    """
    def __init__(self, min_anchors: int = 4, max_anchors: int = 128):
        self.min_anchors = min_anchors
        self.max_anchors = max_anchors
        self.current_anchors = min_anchors

    def adjust(self, retrieval_success: float, sequence_entropy: float):
        """
        Increases anchor density if retrieval fails or entropy is high.
        """
        if retrieval_success < 0.95:
            self.current_anchors = min(self.max_anchors, int(self.current_anchors * 1.5))
        elif retrieval_success > 0.99 and sequence_entropy < 0.5:
            self.current_anchors = max(self.min_anchors, int(self.current_anchors * 0.8))
            
        return self.current_anchors

    def get_anchors(self, scores: torch.Tensor):
        """
        Returns the top-N indices to be used as anchors.
        """
        _, indices = torch.topk(scores, self.current_anchors, dim=-1)
        return indices
