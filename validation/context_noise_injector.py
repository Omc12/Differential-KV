"""
validation/context_noise_injector.py

Phase 12.5D: Context Noise Injector
Injects unstructured noise into the KV cache context to test the resilience
of the sparse attention mechanism against degraded context.
"""

import torch
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory

class ContextNoiseInjector:
    """
    Corrupts background KV states to ensure the system relies correctly
    on the stabilized semantic anchors.
    """
    def inject_noise(self, kv_tensor: torch.Tensor, noise_std: float = 0.5):
        """Adds Gaussian noise to the KV states."""
        noise = torch.randn_like(kv_tensor) * noise_std
        return kv_tensor + noise

    def corrupt_unanchored_regions(self, kv_tensor: torch.Tensor, memory: SemanticAnchorMemory):
        """
        Severely degrades parts of the KV cache that are not protected by anchors.
        """
        corrupted = self.inject_noise(kv_tensor, noise_std=1.0)
        
        # Restore anchored positions
        for pos, anchor in memory.anchors.items():
            if pos < corrupted.shape[0] and anchor.kv_exact is not None:
                # Assuming shape [seq, 2, heads, dim]
                corrupted[pos] = anchor.kv_exact.to(corrupted.device)
                
        return corrupted
