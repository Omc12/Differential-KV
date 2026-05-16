"""
training/geometric_alignment_loss.py

Loss function that aligns head role allocations with geometric manifold properties.
"""

import torch
import torch.nn as nn

class GeometricAlignmentLoss(nn.Module):
    """
    Computes loss based on how well specialized heads 'obey' the geometric state.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        role_probs: torch.Tensor,   # [batch, n_heads, n_roles]
        manifold_stats: torch.Tensor # [batch, 3] (drift, curvature, entropy)
    ) -> torch.Tensor:
        """
        Calculates alignment.
        Example rule: If curvature is high, 'stabilization' role (index 1) 
        should have higher total weight across heads.
        """
        batch_size = role_probs.shape[0]
        curvature = manifold_stats[:, 1]
        drift = manifold_stats[:, 0]
        
        # 1. Stabilization Alignment
        # Weighted sum of stabilization probabilities should correlate with curvature
        stab_probs = role_probs[:, :, 1].mean(dim=1) # [batch]
        stab_loss = torch.mean((stab_probs - torch.sigmoid(curvature))**2)
        
        # 2. Retrieval Alignment
        # If drift is low, 'retrieval' role (index 0) should be dominant
        ret_probs = role_probs[:, :, 0].mean(dim=1) # [batch]
        ret_loss = torch.mean((ret_probs - (1.0 - torch.sigmoid(drift)))**2)
        
        # 3. Diversity Constraint
        # Ensure not all heads pick the same role (unless necessary)
        # (Handled by entropy in the trainer, but can add overlap penalty here)
        
        total_loss = stab_loss + ret_loss
        
        return {
            "total_loss": total_loss,
            "alignment_loss": total_loss,
            "stab_loss": stab_loss,
            "ret_loss": ret_loss
        }
