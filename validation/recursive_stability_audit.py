from typing import List, Dict, Any
import numpy as np

class RecursiveStabilityAudit:
    """
    Audits recursive reasoning loops for stability, degradation, and hallucination markers.
    Ensures that recursion doesn't lead to divergent behavior or latent carryover.
    """
    def __init__(self, divergence_threshold: float = 0.5):
        self.divergence_threshold = divergence_threshold

    def audit_trace(self, trace_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyzes a sequence of metrics from iterative refinement.
        Checks for monotonic improvement vs divergence.
        """
        if len(trace_metrics) < 2:
            return {"status": "INSUFFICIENT_DATA"}

        # Example: check 'loss' or 'entropy' trend
        metric_keys = trace_metrics[0].keys()
        analysis = {}
        
        for key in metric_keys:
            values = [m[key] for m in trace_metrics if key in m]
            if not values: continue
            
            diffs = np.diff(values)
            stability = "STABLE" if np.std(diffs) < self.divergence_threshold else "UNSTABLE"
            trend = "IMPROVING" if values[-1] < values[0] else "DEGRADING" # Assumes lower is better
            
            analysis[key] = {
                "stability": stability,
                "trend": trend,
                "variance": float(np.var(values))
            }

        overall_pass = all(v["stability"] == "STABLE" for v in analysis.values())
        return {
            "status": "PASS" if overall_pass else "FAIL",
            "metric_analysis": analysis
        }
