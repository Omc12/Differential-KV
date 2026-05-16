import torch

class SparseTransitionLinker:
    """
    PHASE 19.0A: Sparse Transition Linker.
    Ensures that symbolic transitions have sufficient 'lead-in' and 'lead-out' 
    continuity to prevent relational collapse.
    """
    def __init__(self, lead_in: int = 8, lead_out: int = 4):
        self.lead_in = lead_in
        self.lead_out = lead_out

    def link_transitions(self, symbolic_mask: torch.Tensor) -> torch.Tensor:
        """
        Expands symbolic tokens to include local transition neighborhoods.
        """
        if not symbolic_mask.any():
            return symbolic_mask
            
        # Use 1D convolution or simple shift-and-or to expand mask
        expanded_mask = symbolic_mask.clone()
        
        # Lead-in (tokens before symbolic)
        for i in range(1, self.lead_in + 1):
            shifted = torch.cat([torch.zeros((1, i), dtype=torch.bool, device=symbolic_mask.device), 
                               symbolic_mask[:, :-i]], dim=1)
            expanded_mask |= shifted
            
        # Lead-out (tokens after symbolic)
        for i in range(1, self.lead_out + 1):
            shifted = torch.cat([symbolic_mask[:, i:], 
                               torch.zeros((1, i), dtype=torch.bool, device=symbolic_mask.device)], dim=1)
            expanded_mask |= shifted
            
        return expanded_mask
