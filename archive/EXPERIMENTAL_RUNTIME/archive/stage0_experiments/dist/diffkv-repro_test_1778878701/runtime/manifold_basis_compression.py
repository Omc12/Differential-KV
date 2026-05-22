"""
runtime/manifold_basis_compression.py

Compresses manifold vectors into a set of basis vectors.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ManifoldBasisCompression:
    """
    Representing the manifold using a sparse basis.
    """
    def __init__(self, head_dim: int, n_basis: int = 16):
        self.head_dim = head_dim
        self.n_basis = n_basis

    def extract_basis(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uses SVD to extract the most important basis vectors.
        vectors: [batch, n_heads, seq_len, head_dim]
        """
        B, H, S, D = vectors.shape
        
        # Flatten for SVD: [B*H, S, D]
        x = vectors.view(B * H, S, D)
        
        # SVD
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        
        # Take top n_basis
        basis = vh[:, :self.n_basis, :] # [B*H, n_basis, D]
        coefficients = u[:, :, :self.n_basis] * s[:, :self.n_basis].unsqueeze(1) # [B*H, S, n_basis]
        
        return basis.view(B, H, self.n_basis, D), coefficients.view(B, H, S, self.n_basis)

    def reconstruct(self, basis: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs original vectors.
        """
        return torch.matmul(coefficients, basis)

if __name__ == "__main__":
    B, H, S, D = 1, 8, 256, 64
    mbc = ManifoldBasisCompression(D, n_basis=16)
    
    vectors = torch.randn(B, H, S, D)
    basis, coeffs = mbc.extract_basis(vectors)
    recon = mbc.reconstruct(basis, coeffs)
    
    error = torch.norm(vectors - recon) / torch.norm(vectors)
    print(f"Basis Reconstruction Error (Normalized): {error.item():.4f}")
    print(f"Compression Ratio: {vectors.numel() / (basis.numel() + coeffs.numel()):.2f}x")
