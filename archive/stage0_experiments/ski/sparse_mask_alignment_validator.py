
import torch
from typing import Dict, Any

class SparseMaskAlignmentValidator:
    """
    PHASE 24.5: Sparse Mask Alignment Validator (SKI).
    Detects drift between sparse mask execution and logical symbolic intent.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alignment_scores = []
        
    def validate_alignment(self, 
                           sparse_mask: torch.Tensor, 
                           logical_intent: torch.Tensor) -> float:
        """
        Calculates the alignment between the hardware-ready sparse mask 
        and the high-level symbolic routing intent.
        """
        # Logical AND / OR check for alignment
        intersection = (sparse_mask.bool() & logical_intent.bool()).float().sum()
        union = (sparse_mask.bool() | logical_intent.bool()).float().sum()
        
        iou = (intersection / (union + 1e-9)).item()
        self.alignment_scores.append(iou)
        
        return iou

    def get_alignment_metrics(self) -> Dict[str, float]:
        avg_iou = sum(self.alignment_scores) / len(self.alignment_scores) if self.alignment_scores else 1.0
        return {
            "sparse_mask_integrity": avg_iou,
            "mask_drift_detected": avg_iou < 0.99
        }
