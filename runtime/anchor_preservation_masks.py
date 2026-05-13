import torch

def create_anchor_mask(seq_len: int, sink_size: int, anchors: torch.Tensor, device: torch.device):
    """
    Creates a preservation mask for sinks and anchors.
    anchors: Tensor of indices to preserve.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    
    # Sinks
    if sink_size > 0:
        mask[:min(seq_len, sink_size)] = True
        
    # Anchors
    if anchors is not None and anchors.numel() > 0:
        valid_anchors = anchors[anchors < seq_len]
        mask[valid_anchors] = True
        
    return mask

def apply_anchor_mask(attention_scores: torch.Tensor, mask: torch.Tensor, value: float = -1e4):
    """
    Apply mask to attention scores to protect certain tokens from being ignored.
    Or more likely, used for pruning to ensure these aren't pruned.
    """
    # In context of pruning, we want to keep these tokens.
    # If this is for attention masking, usually we mask out tokens.
    # Here 'mask' means 'preserve'.
    return attention_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), value)
