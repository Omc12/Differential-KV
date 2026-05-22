from .density_drift_monitor import DensityDriftMonitor
from .sparse_oscillation_detector import SparseOscillationDetector
from .retrieval_decay_controller import RetrievalDecayController

class LongHorizonSparseStability:
    """
    Coordinates sparse stability tests over long durations.
    """
    def __init__(self):
        self.density_monitor = DensityDriftMonitor()
        self.oscillation_detector = SparseOscillationDetector()
        self.decay_controller = RetrievalDecayController()

    def audit_step(self, density: float, retrieval_score: float):
        self.density_monitor.log_density(density)
        self.oscillation_detector.log_density(density)
        
        if self.decay_controller.should_adjust(retrieval_score):
            return self.decay_controller.get_adjustment()
        return None

    def get_stability_report(self):
        return {
            "density_drift": self.density_monitor.get_drift(),
            "oscillations": self.oscillation_detector.get_oscillation_count(),
            "stability_status": "STABLE" if self.density_monitor.get_drift() < 0.1 else "DRIFTING"
        }
