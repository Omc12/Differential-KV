import torch
import numpy as np
from typing import Dict, Any

class TemporalPhaseMemoryAnalysis:
    """
    Measures how well temporal phase structure is preserved across reasoning steps.
    """
    def __init__(self):
        self.phase_errors = []
        self.rhythm_consistency = []

    def log_phase(self, expected_phase: int, actual_latent: torch.Tensor, phase_anchors: torch.Tensor):
        # actual_latent: [d_model]
        # phase_anchors: [cadence_period, d_model]
        
        # Find which anchor the current latent is closest to
        similarities = torch.nn.functional.cosine_similarity(
            actual_latent.unsqueeze(0), phase_anchors, dim=1
        )
        inferred_phase = torch.argmax(similarities).item()
        
        error = abs(inferred_phase - expected_phase)
        self.phase_errors.append(error)
        self.rhythm_consistency.append(1.0 if error == 0 else 0.0)

    def get_metrics(self) -> Dict[str, float]:
        return {
            "mean_phase_error": np.mean(self.phase_errors) if self.phase_errors else 0,
            "rhythm_fidelity": np.mean(self.rhythm_consistency) if self.rhythm_consistency else 0
        }
