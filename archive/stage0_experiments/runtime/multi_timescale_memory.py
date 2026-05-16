import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

class MultiTimescaleMemory:
    """
    PHASE 25: Multi-Timescale Cognitive Memory
    Manages fast (token-level), medium (reasoning block), and slow (global narrative) memory scales.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        
        # Memory buffers
        self.fast_buffer = torch.zeros(d_model) # Token level
        self.medium_buffer = torch.zeros(d_model) # Block level (e.g., 32 tokens)
        self.slow_buffer = torch.zeros(d_model) # Global narrative
        
        # Decay rates
        self.fast_decay = 0.1
        self.medium_decay = 0.95
        self.slow_decay = 0.999
        
        self.step_counter = 0

    def update(self, current_latent: torch.Tensor):
        """
        Updates memory buffers at multiple timescales.
        """
        self.step_counter += 1
        device = current_latent.device
        
        # Move buffers to device if needed
        self.fast_buffer = self.fast_buffer.to(device)
        self.medium_buffer = self.medium_buffer.to(device)
        self.slow_buffer = self.slow_buffer.to(device)
        
        # 1. Fast: Immediate stabilization (residual-like)
        self.fast_buffer = current_latent.detach()
        
        # 2. Medium: Reasoning coherence
        self.medium_buffer = self.medium_decay * self.medium_buffer + (1 - self.medium_decay) * current_latent.detach()
        
        # 3. Slow: Global narrative persistence
        self.slow_buffer = self.slow_decay * self.slow_buffer + (1 - self.slow_decay) * current_latent.detach()

    def get_memory_signal(self) -> torch.Tensor:
        """
        Returns a hierarchical memory signal to stabilize inference.
        """
        # Weighted combination of timescales
        # Fast gives local continuity, medium gives block coherence, slow gives narrative global structure
        return (0.1 * self.fast_buffer + 
                0.3 * self.medium_buffer + 
                0.6 * self.slow_buffer)

    def get_hierarchy_stats(self) -> Dict[str, float]:
        return {
            "fast_norm": self.fast_buffer.norm().item(),
            "medium_norm": self.medium_buffer.norm().item(),
            "slow_norm": self.slow_buffer.norm().item(),
            "temporal_ratio": self.medium_buffer.norm().item() / (self.slow_buffer.norm().item() + 1e-6)
        }
