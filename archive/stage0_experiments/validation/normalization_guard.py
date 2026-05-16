import numpy as np
from typing import List, Union
try:
    from .metric_range_assertions import assert_not_nan_inf
except ImportError:
    try:
        from validation.metric_range_assertions import assert_not_nan_inf
    except ImportError:
        from metric_range_assertions import assert_not_nan_inf

class NormalizationGuard:
    """
    Ensures metrics are correctly normalized and aggregated without overflow.
    """
    
    @staticmethod
    def verify_probability_distribution(probs: Union[List[float], np.ndarray], name: str = "probs", tol: float = 1e-6):
        """
        Verify that a sequence of probabilities sums to 1.0.
        """
        probs_array = np.array(probs)
        for val in probs_array:
            assert_not_nan_inf(val, f"{name}_element")
        
        sum_val = np.sum(probs_array)
        if abs(sum_val - 1.0) > tol:
            raise ValueError(f"CRITICAL ERROR: Distribution '{name}' not normalized. Sum: {sum_val}")

    @staticmethod
    def audit_aggregation(metrics: List[float], name: str = "aggregated_metric"):
        """
        Audits an aggregated metric for variance and confidence intervals.
        Ensures no silent collapse or overflow.
        """
        if not metrics:
            return {"mean": 0.0, "std": 0.0, "ci_95": 0.0}
            
        arr = np.array(metrics)
        for val in arr:
            assert_not_nan_inf(val, name)
            
        mean = np.mean(arr)
        std = np.std(arr)
        # Simple 95% CI assuming normal distribution
        ci_95 = 1.96 * (std / np.sqrt(len(arr)))
        
        # Detect suspicious variance
        if std < 1e-9 and len(arr) > 1:
            print(f"WARNING: Zero variance detected in '{name}'. Possible data contamination or fixed results.")
            
        return {
            "mean": mean,
            "std": std,
            "ci_95": ci_95,
            "count": len(arr)
        }

    @staticmethod
    def detect_overflow(value: float, threshold: float = 1e15, name: str = "metric"):
        """
        Detect potential aggregation overflows before they hit Inf.
        """
        if abs(value) > threshold:
            raise ValueError(f"CRITICAL ERROR: Potential overflow detected in '{name}': {value}")

if __name__ == "__main__":
    print("Running NormalizationGuard self-test...")
    guard = NormalizationGuard()
    
    # Test probability distribution
    guard.verify_probability_distribution([0.2, 0.3, 0.5])
    print("[PASS] Probability distribution check passed")
    
    # Test aggregation
    stats = guard.audit_aggregation([0.8, 0.85, 0.82, 0.88], "test_metric")
    print(f"[PASS] Aggregation stats: {stats}")
    
    try:
        guard.verify_probability_distribution([0.5, 0.6])
    except ValueError as e:
        print(f"[PASS] Caught unnormalized sum: {e}")
        
    print("NormalizationGuard validated.")
