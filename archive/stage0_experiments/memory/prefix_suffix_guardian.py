import torch

class PrefixSuffixGuardian:
    """
    PHASE 18.9C: Prefix/Suffix Guardian.
    Specifically targets the edges of symbolic spans for high-precision retention.
    Ensures that "IDENTIFIER-" and "-SIGMA" edges survive pruning.
    """
    def __init__(self, edge_size=8):
        self.edge_size = edge_size

    def protect_edges(self, symbolic_spans, seq_len):
        """
        Returns a mask for symbolic edges.
        """
        edge_mask = torch.zeros(seq_len, dtype=torch.bool)
        for start, end in symbolic_spans:
            # 1. Prefix protection (Lead-in)
            # We protect a small window at the start of the span
            p_start = max(0, start)
            p_end = min(seq_len, start + self.edge_size)
            edge_mask[p_start:p_end] = True
            
            # 2. Suffix protection (Lead-out)
            # We protect a small window at the end of the span
            s_start = max(0, end - self.edge_size + 1)
            s_end = min(seq_len, end + 1)
            edge_mask[s_start:s_end] = True
            
            # 3. Transition protection
            # Also protect 1-2 tokens BEFORE the start and AFTER the end
            # to ensure the boundary itself is clean.
            t_start = max(0, start - 2)
            t_end = min(seq_len, end + 3)
            edge_mask[t_start:start] = True
            edge_mask[end+1:t_end] = True
            
        return edge_mask
