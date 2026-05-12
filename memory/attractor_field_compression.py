"""
memory/attractor_field_compression.py

Compresses attractor fields into ultra-sparse representations.
"""

import torch
import torch.nn as nn
from typing import Tuple

class AttractorFieldCompression:
    """
    Uses vector quantization to compress attractor fields.
    """
    def __init__(self, head_dim: int, codebook_size: int = 256):
        self.head_dim = head_dim
        self.codebook_size = codebook_size
        self.codebook = nn.Parameter(torch.randn(codebook_size, head_dim))

    def compress(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses field [H, N, D] into indices [H, N] and residuals.
        """
        H, N, D = field.shape
        field_flat = field.view(-1, D)
        
        # Calculate distances to codebook
        dists = torch.cdist(field_flat, self.codebook) # [H*N, CB]
        indices = torch.argmin(dists, dim=-1)
        
        # Calculate residuals
        quantized = self.codebook[indices]
        residuals = field_flat - quantized
        
        return indices.view(H, N), residuals.view(H, N, D)

    def decompress(self, indices: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
        """
        Decompresses back to [H, N, D].
        """
        quantized = self.codebook[indices]
        return quantized + residuals

if __name__ == "__main__":
    H, N, D = 8, 128, 64
    afc = AttractorFieldCompression(D)
    
    field = torch.randn(H, N, D)
    indices, residuals = afc.compress(field)
    decompressed = afc.decompress(indices, residuals)
    
    recon_error = torch.norm(field - decompressed)
    print(f"Reconstruction Error: {recon_error.item():.4e}")
    print(f"Compression Complete.")
