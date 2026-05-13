import torch

class ContextEntropyScheduler:
    """
    Schedules sparsity pressure based on total context length and VRAM pressure.
    Prevents retrieval collapse by relaxing sparsity during complex reasoning.
    """
    def __init__(self, max_context: int = 128000, min_sparsity: float = 0.0, max_sparsity: float = 0.8):
        self.max_context = max_context
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        
    def get_current_sparsity(self, current_len: int, vram_pressure: float = 0.0) -> float:
        """
        Calculates target sparsity.
        vram_pressure: 0.0 to 1.0 (normalized)
        """
        # Linear scaling of sparsity based on context length
        length_factor = min(current_len / self.max_context, 1.0)
        
        # Base sparsity increases with length
        base_sparsity = self.min_sparsity + (self.max_sparsity - self.min_sparsity) * length_factor
        
        # VRAM pressure can force higher sparsity
        final_sparsity = max(base_sparsity, vram_pressure * self.max_sparsity)
        
        return torch.clamp(torch.tensor(final_sparsity), self.min_sparsity, self.max_sparsity).item()

    def adjust_density(self, current_len: int) -> float:
        """Returns the density (1 - sparsity)."""
        return 1.0 - self.get_current_sparsity(current_len)
