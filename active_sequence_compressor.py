import torch
from typing import Tuple

class ActiveSequenceCompressor:
    """
    Handles dynamic sequence collapse and token gather/scatter.
    Reduces the number of tokens that traverse the compute paths.
    """
    def __init__(self):
        pass

    def compress_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers active tokens based on the survival mask.
        x: [bsz, seq_len, d]
        mask: [bsz, seq_len]
        """
        bsz, seq_len, d = x.shape
        # Note: In a real implementation, we handle variable sequence lengths per batch
        # For simulation, we assume uniform sparsity
        
        # Simplified gather for infrastructure
        active_indices = torch.where(mask[0])[0] # Assuming bsz=1 or same mask
        compressed_x = x[:, active_indices, :]
        
        return compressed_x, active_indices

    def decompress_sequence(self, compressed_x: torch.Tensor, indices: torch.Tensor, original_len: int) -> torch.Tensor:
        """
        Scatters active tokens back into the full sequence.
        """
        bsz, comp_len, d = compressed_x.shape
        full_x = torch.zeros((bsz, original_len, d), device=compressed_x.device, dtype=compressed_x.dtype)
        full_x[:, indices, :] = compressed_x
        return full_x
