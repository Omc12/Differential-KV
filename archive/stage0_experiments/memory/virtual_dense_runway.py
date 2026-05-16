import torch

class VirtualDenseRunway:
    """
    PHASE 19.0B: Virtual Dense Runway.
    Creates a temporary high-density attention zone around active transitions
    to stabilize symbolic lead-ins.
    """
    def __init__(self, runway_width: int = 32, decay_rate: float = 0.5):
        self.runway_width = runway_width
        self.decay_rate = decay_rate
        self.active_runways = [] # List of (center_idx, width)

    def apply_runway(self, importance_scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        Boosts importance scores in the neighborhood of target indices to 
        create a 'runway' effect.
        """
        if target_indices.numel() == 0:
            return importance_scores
            
        seq_len = importance_scores.shape[1]
        boosted_importance = importance_scores.clone()
        
        for idx in target_indices:
            start = max(0, int(idx) - self.runway_width // 2)
            end = min(seq_len, int(idx) + self.runway_width // 2)
            
            if start >= end:
                continue
            
            # Create a smooth gradient boost
            # Tokens closer to idx get higher boost
            neighborhood_indices = torch.arange(start, end, device=importance_scores.device)
            distances = torch.abs(neighborhood_indices - idx)
            boost = torch.exp(-self.decay_rate * distances.float()) * 1000.0 # Scale boost
            boost = boost.to(importance_scores.dtype)
            
            boosted_importance[0, start:end] += boost
            
        return boosted_importance
