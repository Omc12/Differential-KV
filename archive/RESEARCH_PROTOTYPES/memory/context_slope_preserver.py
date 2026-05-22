import torch

class ContextSlopePreserver:
    """
    PHASE 19.0C: Context Slope Preserver.
    Enforces a minimum importance slope between distant symbolic regions 
    to maintain a 'traversable' landscape.
    """
    def __init__(self, min_slope: float = 0.01):
        self.min_slope = min_slope

    def preserve_slopes(self, importance_scores: torch.Tensor, absolute_indices: torch.Tensor, symbolic_indices: torch.Tensor) -> torch.Tensor:
        """
        Adjusts importance scores between symbolic points that are present 
        in the current cache.
        """
        if symbolic_indices.numel() < 2 or absolute_indices is None:
            return importance_scores
            
        seq_len = importance_scores.shape[1]
        adjusted_scores = importance_scores.clone()
        
        # Find which symbolic indices are present in the current absolute_indices
        # absolute_indices is [batch, seq_len]
        # symbolic_indices is [num_sym]
        
        # This is a bit slow but robust: find local indices
        # We only care about the first batch for simplicity in this sparse context
        abs_indices_list = absolute_indices[0].tolist()
        local_sym_indices = []
        for sym_idx in symbolic_indices.tolist():
            try:
                local_idx = abs_indices_list.index(int(sym_idx))
                local_sym_indices.append(local_idx)
            except ValueError:
                continue
        
        if len(local_sym_indices) < 2:
            return importance_scores
            
        local_sym_indices.sort()
        
        for i in range(len(local_sym_indices) - 1):
            start = local_sym_indices[i]
            end = local_sym_indices[i+1]
            
            # Linear interpolation for minimum floor
            floor = torch.linspace(1.0, 1.0, steps=end-start, device=importance_scores.device, dtype=importance_scores.dtype) * self.min_slope
            adjusted_scores[0, start:end] = torch.maximum(adjusted_scores[0, start:end], floor)
            
        return adjusted_scores
