import torch

class AttentionSinkPreserver:
    """
    Ensures that the first N tokens (attention sinks) are never evicted or pruned.
    """
    def __init__(self, sink_size=4):
        self.sink_size = sink_size

    def get_protected_indices(self, seq_len):
        return list(range(min(seq_len, self.sink_size)))

    def apply_protection(self, importance_scores):
        """
        Set importance of sink tokens to infinity.
        importance_scores: [..., seq_len]
        """
        if importance_scores.size(-1) > self.sink_size:
            importance_scores[..., :self.sink_size] = float('inf')
        return importance_scores
