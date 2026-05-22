import torch
from robustness.adversarial_geometry_detector import AdversarialGeometryDetector
from robustness.resonance_reanchoring import ResonanceReanchoring

class ActiveManifoldHardening:
    """
    Coordinates adversarial drift detection and recovery.
    Provides anti-collapse stabilization.
    """
    def __init__(self, d_model: int):
        self.detector = AdversarialGeometryDetector(d_model)
        self.reanchorer = ResonanceReanchoring(d_model)
        self.attack_history = []
        
    def harden_manifold(self, manifold_states: torch.Tensor, reference_manifold: torch.Tensor) -> torch.Tensor:
        """
        Monitors for attacks and applies recovery if needed.
        """
        report = self.detector.detect_adversarial_drift(manifold_states, reference_manifold)
        
        if report["is_adversarial"]:
            self.attack_history.append(report)
            return self.reanchorer.reanchor_manifold(manifold_states)
        else:
            if report["drift_magnitude"] < 0.05:
                self.reanchorer.add_stable_anchor(manifold_states.mean(dim=1))
            return manifold_states

    def get_hardening_report(self) -> dict:
        return {
            "total_attacks_detected": len(self.attack_history),
            "avg_attack_severity": sum(a["risk_score"] for a in self.attack_history) / (len(self.attack_history) + 1e-6),
            "current_robustness_level": 1.0 - (len(self.attack_history) / 1000.0)
        }
