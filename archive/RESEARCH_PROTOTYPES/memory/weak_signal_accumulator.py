import torch

class WeakSignalAccumulator:
    """
    PHASE 20.1C: Accumulates weak symbolic evidence across contexts.
    Allows low-variance identifiers to eventually trigger preservation 
    if they appear persistently or in proximity to other weak signals.
    """
    def __init__(self, persistence_threshold: float = 2.0):
        self.persistence_threshold = persistence_threshold
        self.accumulation_buffer = {} # {abs_index: accumulated_score}

    def accumulate(self, indices: torch.Tensor, low_salience_scores: torch.Tensor):
        """
        Adds evidence for indices that didn't meet the primary threshold.
        """
        flat_indices = indices.flatten().tolist()
        flat_scores = low_salience_scores.flatten().tolist()
        
        for idx, score in zip(flat_indices, flat_scores):
            # Accumulate scores for persistent weak signals
            self.accumulation_buffer[idx] = self.accumulation_buffer.get(idx, 0.0) + score
            
    def get_accumulation_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask of indices that have accumulated enough evidence to be 'upgraded'.
        """
        mask = torch.zeros_like(indices, dtype=torch.bool)
        flat_indices = indices.flatten().tolist()
        for i, idx in enumerate(flat_indices):
            if self.accumulation_buffer.get(idx, 0.0) >= self.persistence_threshold:
                mask.flatten()[i] = True
        return mask

    def flush_old_signals(self, current_global_offset: int, window: int = 32768):
        """
        Removes very old signals to prevent memory growth.
        """
        self.accumulation_buffer = {k: v for k, v in self.accumulation_buffer.items() if k > current_global_offset - window}
