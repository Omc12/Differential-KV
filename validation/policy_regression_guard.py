from typing import Dict, Any, List

class PolicyRegressionGuard:
    """
    Detects performance regressions after policy updates.
    Ensures that "optimizations" don't degrade core metrics.
    """
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance
        self.baseline_metrics = {}

    def set_baseline(self, metrics: Dict[str, float]):
        """Sets the baseline metrics for comparison."""
        self.baseline_metrics = metrics

    def check_regression(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Checks if current metrics have regressed compared to baseline.
        Returns a report of regressions.
        """
        if not self.baseline_metrics:
            return {"status": "NO_BASELINE"}

        regressions = []
        for key, baseline_val in self.baseline_metrics.items():
            if key not in current_metrics: continue
            
            current_val = current_metrics[key]
            # Assumes lower is better for latency, memory; higher is better for accuracy
            is_regression = False
            if key in ["latency", "memory_usage"]:
                if current_val > baseline_val * (1 + self.tolerance):
                    is_regression = True
            elif key in ["accuracy", "retrieval_efficiency"]:
                if current_val < baseline_val * (1 - self.tolerance):
                    is_regression = True
                    
            if is_regression:
                regressions.append({
                    "metric": key,
                    "baseline": baseline_val,
                    "current": current_val,
                    "change": (current_val - baseline_val) / baseline_val
                })

        return {
            "status": "PASS" if not regressions else "FAIL",
            "regressions": regressions
        }
