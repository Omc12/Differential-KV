import math
from typing import Any, Optional

def assert_not_nan_inf(value: Any, name: str = "metric"):
    """
    Assert that a metric is not NaN or Infinity.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric '{name}' must be a number, got {type(value)}")
    
    if math.isnan(value):
        raise ValueError(f"CRITICAL ERROR: Metric '{name}' is NaN")
    
    if math.isinf(value):
        raise ValueError(f"CRITICAL ERROR: Metric '{name}' is Infinite")

def assert_metric_in_range(value: float, min_val: float, max_val: float, name: str = "metric"):
    """
    Assert that a metric is within a specific range [min_val, max_val].
    """
    assert_not_nan_inf(value, name)
    
    if value < min_val or value > max_val:
        raise ValueError(f"CRITICAL ERROR: Metric '{name}' ({value}) out of bounds [{min_val}, {max_val}]")

def validate_retrieval_score(score: float):
    """
    Standard validation for retrieval scores [0.0, 1.0].
    """
    assert_metric_in_range(score, 0.0, 1.0, "retrieval_score")

def validate_retention_percent(percent: float):
    """
    Standard validation for retention percentage [0.0, 100.0].
    """
    assert_metric_in_range(percent, 0.0, 100.0, "retention_percent")

if __name__ == "__main__":
    # Self-test
    print("Running metric range assertions self-test...")
    try:
        validate_retrieval_score(0.85)
        validate_retention_percent(42.0)
        print("[PASS] Basic range tests passed")
        
        try:
            validate_retrieval_score(1.1)
        except ValueError as e:
            print(f"[PASS] Caught expected overflow: {e}")
            
        try:
            assert_not_nan_inf(float('nan'))
        except ValueError as e:
            print(f"[PASS] Caught expected NaN: {e}")
            
    except Exception as e:
        print(f"TEST FAILED: {e}")
        exit(1)
    print("Metric range assertions validated.")
