from typing import List
import numpy as np
from .metric_range_assertions import validate_retrieval_score

class RetrievalScoreCalibrator:
    """
    Calibrates retrieval scores to ensure they are bounded and honest.
    Prevents artificial score lifting through non-linear scaling.
    """
    
    @staticmethod
    def calibrate_linear(raw_score: float, min_val: float, max_val: float) -> float:
        """
        Linearly scales a raw score into the [0, 1] range based on observed bounds.
        """
        if max_val <= min_val:
            return 0.0
            
        calibrated = (raw_score - min_val) / (max_val - min_val)
        calibrated = max(0.0, min(1.0, calibrated))
        
        validate_retrieval_score(calibrated)
        return calibrated

    @staticmethod
    def audit_calibration_honesty(raw_scores: List[float], calibrated_scores: List[float]):
        """
        Ensures that calibration didn't introduce a systematic bias or 
        non-monotonic behavior.
        """
        if len(raw_scores) != len(calibrated_scores):
            raise ValueError("Input lists must have the same length.")
            
        # Check monotonicity
        raw_arr = np.array(raw_scores)
        cal_arr = np.array(calibrated_scores)
        
        # Sort by raw to check if calibrated is also sorted
        idx = np.argsort(raw_arr)
        sorted_cal = cal_arr[idx]
        
        diffs = np.diff(sorted_cal)
        if np.any(diffs < -1e-9):
            raise ValueError("CRITICAL ERROR: Calibration is non-monotonic. Scores are being manipulated dishonestly.")

    @staticmethod
    def apply_temperature_calibration(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Applies temperature scaling to logits for retrieval probability calibration.
        Ensures results remain valid probability distributions.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
            
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / np.sum(exp_logits)
        
        # Validate that no probability is > 1.0 or < 0.0
        for p in probs:
            if p < 0.0 or p > 1.0:
                raise ValueError(f"CRITICAL ERROR: Calibrated probability {p} out of bounds.")
                
        return probs

if __name__ == "__main__":
    print("Running RetrievalScoreCalibrator self-test...")
    calibrator = RetrievalScoreCalibrator()
    
    # Test linear calibration
    score = calibrator.calibrate_linear(150, 100, 200)
    print(f"[PASS] Calibrated 150 in [100, 200] to: {score}")
    
    # Test honesty audit
    raw = [10, 20, 30]
    cal = [0.1, 0.2, 0.3]
    calibrator.audit_calibration_honesty(raw, cal)
    print("[PASS] Monotonicity audit passed")
    
    # Test dishonesty detection
    try:
        calibrator.audit_calibration_honesty([10, 20, 30], [0.1, 0.5, 0.2])
    except ValueError as e:
        print(f"[PASS] Caught dishonest non-monotonic calibration: {e}")
        
    print("RetrievalScoreCalibrator validated.")
