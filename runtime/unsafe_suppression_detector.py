from typing import Dict, Any, Optional

class UnsafeSuppressionDetector:
    """
    SGC Phase 39.1 RESET: Unsafe Suppression Detector.
    Detects cases where governance prevented fallback but semantic quality degraded.
    """
    def __init__(self, drift_threshold: float = 0.05):
        self.drift_threshold = drift_threshold

    def evaluate_suppression(self, 
                             suppressed: bool, 
                             confidence: float, 
                             drift_score: float) -> Dict[str, Any]:
        """
        Analyzes a suppression event for safety.
        """
        is_unsafe = False
        reason = None
        
        if suppressed and drift_score > self.drift_threshold:
            is_unsafe = True
            reason = f"High drift ({drift_score:.4f}) despite suppression (conf={confidence:.4f})"
        elif not suppressed and drift_score < (self.drift_threshold / 2):
            # This is "Excessive Fallback" - opposite of unsafe suppression, 
            # but still useful for efficiency tuning.
            pass

        return {
            "is_unsafe_suppression": is_unsafe,
            "suppression_risk": drift_score if suppressed else 0.0,
            "reason": reason
        }
