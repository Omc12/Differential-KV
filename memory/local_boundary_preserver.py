class LocalBoundaryPreserver:
    """
    PHASE 18.9B: Local Boundary Preserver.
    Ensures that structural boundaries (anchors) are physically pinned in the KV cache.
    """
    def __init__(self, neighborhood=8):
        self.neighborhood = neighborhood

    def get_protection_mask(self, anchor_indices, seq_len):
        import torch
        mask = torch.zeros(seq_len, dtype=torch.bool)
        for idx in anchor_indices:
            start = max(0, idx - self.neighborhood)
            end = min(seq_len - 1, idx + self.neighborhood)
            mask[start:end+1] = True
        return mask
