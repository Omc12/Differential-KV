import torch
from typing import Dict, Any, Tuple
from .manifold_identity_anchors import ManifoldIdentityAnchors
from .cognitive_integrity_monitor import CognitiveIntegrityMonitor

class IdentityDriftController:
    """
    Actively regulates and corrects identity drift.
    Uses anchoring forces and manifold re-centering to maintain stability.
    """
    def __init__(self, anchors: ManifoldIdentityAnchors, monitor: CognitiveIntegrityMonitor):
        self.anchors = anchors
        self.monitor = monitor
        self.correction_strength = 0.05
        self.total_corrections = 0

    def regulate_drift(self, manifolds: torch.Tensor, current_fp: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Analyzes drift and applies necessary corrections to the manifolds.
        """
        integrity = self.monitor.check_integrity(current_fp, manifolds)
        
        corrected_manifolds = manifolds
        applied_correction = False
        
        # If identity similarity is dropping, increase anchoring force
        if integrity["identity_similarity"] < 0.9:
            force = self.correction_strength * (1.0 - integrity["identity_similarity"])
            corrected_manifolds = self.anchors.apply_anchoring_force(manifolds, strength=force)
            applied_correction = True
            self.total_corrections += 1
            
        # If runaway divergence is detected, perform a drastic re-centering
        if self.monitor.detect_runaway_divergence():
            print("CRITICAL: Runaway divergence detected! Applying emergency re-centering.")
            # Simple re-centering: mix with anchors more strongly
            corrected_manifolds = self.anchors.apply_anchoring_force(corrected_manifolds, strength=0.5)
            applied_correction = True
            
        return corrected_manifolds, {
            "applied_correction": applied_correction,
            "integrity_status": integrity["status"],
            "similarity": integrity["identity_similarity"]
        }

    def set_correction_strength(self, strength: float):
        self.correction_strength = strength
