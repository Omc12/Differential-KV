"""
runtime/attractor_compressor.py
Phase 26: Cognitive Energy Minimization (CEM)
Compresses stable attractors directly to reduce memory overhead while preserving manifold shape.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple

class AttractorCompressor:
    def __init__(self, target_dim: int = 64):
        self.target_dim = target_dim
        self.projection_matrix = None # Learned or random projection

    def compress(self, attractor_state: torch.Tensor, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compresses an attractor into a compact representation.
        Stores the centroid, resonance signature, and geometric boundaries.
        """
        state_flat = attractor_state.detach().cpu().float().numpy().flatten()
        d_model = state_flat.size
        
        # 1. Signature generation (Mean and variance as a simple proxy)
        signature = {
            "mean": float(np.mean(state_flat)),
            "std": float(np.std(state_flat)),
            "max_abs": float(np.max(np.abs(state_flat)))
        }
        
        # 2. Centroid compression
        # For Phase 26, we store a downsampled or quantized version of the centroid
        # Here: simple 10x downsampling as a placeholder for structured compression
        downsampled_centroid = state_flat[::max(1, d_model // self.target_dim)]
        
        compressed = {
            "centroid_compressed": downsampled_centroid,
            "original_dim": d_model,
            "resonance_signature": signature,
            "phase_state": metadata.get("phase_state", 0.0),
            "drift_boundaries": metadata.get("drift_boundaries", 0.15)
        }
        return compressed

    def reconstruct(self, compressed: Dict[str, Any], original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Reconstructs a rough approximation of the attractor from compressed data.
        Used for restoring stability when the runtime drifts.
        """
        compressed_centroid = compressed["centroid_compressed"]
        original_dim = compressed["original_dim"]
        
        # Simple interpolation to reconstruct shape
        xp = np.linspace(0, 1, len(compressed_centroid))
        x = np.linspace(0, 1, original_dim)
        reconstructed_flat = np.interp(x, xp, compressed_centroid)
        
        # Scale by original signature if available
        sig = compressed["resonance_signature"]
        # (Normalization/denormalization step could go here)
        
        return torch.from_numpy(reconstructed_flat).reshape(original_shape).float()
