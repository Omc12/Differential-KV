import torch

class SparseAttentionBridges:
    """
    PHASE 19.0D: Sparse Attention Bridges.
    Implements the structural logic for bridge token selection and 
    activation during inference.
    """
    def __init__(self, bridge_width: int = 4):
        self.bridge_width = bridge_width

    def build_bridges(self, symbolic_indices: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Constructs a sparse bridge mask between consecutive symbolic points.
        """
        if symbolic_indices.numel() < 2:
            return torch.zeros((1, seq_len), dtype=torch.bool, device=symbolic_indices.device)
            
        bridge_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=symbolic_indices.device)
        symbolic_indices, _ = torch.sort(symbolic_indices)
        
        for i in range(symbolic_indices.numel() - 1):
            start = symbolic_indices[i].item()
            end = symbolic_indices[i+1].item()
            
            # Mid-bridge point
            mid = (start + end) // 2
            
            # Create a small island at the midpoint
            b_start = max(0, mid - self.bridge_width // 2)
            b_end = min(seq_len, mid + self.bridge_width // 2)
            bridge_mask[0, b_start:b_end] = True
            
        return bridge_mask
