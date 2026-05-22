
import torch
from typing import Dict, List, Any

class CoordinationFeedbackLoop:
    """
    PHASE 22.4: ARO - Coordination Feedback Loop.
    Tracks delegation outcomes and coordination quality for self-optimization.
    """
    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.feedback_buffer: List[float] = []
        self.coordination_quality = 1.0

    def record_outcome(self, 
                       delegation_success: bool, 
                       arbitration_stability: float):
        """
        Calculates a coordination quality score from delegation and arbitration signals.
        """
        score = (1.0 if delegation_success else 0.5) * arbitration_stability
        self.feedback_buffer.append(score)
        
        if len(self.feedback_buffer) > self.window_size:
            self.feedback_buffer.pop(0)
            
        self.coordination_quality = sum(self.feedback_buffer) / len(self.feedback_buffer)

    def get_feedback_signal(self) -> float:
        return self.coordination_quality

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "coordination_feedback_quality": self.coordination_quality,
            "feedback_variance": torch.tensor(self.feedback_buffer).std().item() if len(self.feedback_buffer) > 1 else 0.0
        }
