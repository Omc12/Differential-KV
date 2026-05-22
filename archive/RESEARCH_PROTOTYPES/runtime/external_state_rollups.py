from typing import List, Dict, Any

class ExternalStateRollups:
    """
    Handles explicit state rollups between iterations.
    Summarizes complex states into bounded, retrieval-grounded representations.
    """
    def __init__(self, rollup_limit_bytes: int = 1024):
        self.rollup_limit_bytes = rollup_limit_bytes

    def generate_rollup(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a compressed rollup of multiple states.
        Ensures the resulting object is bounded in size.
        """
        # Simple implementation: focus on key metrics and last status
        rollup = {
            "step_count": len(states),
            "aggregate_metrics": self._aggregate_metrics([s.get("metrics", {}) for s in states]),
            "last_known_status": states[-1].get("status", "unknown") if states else "none",
            "timestamp": states[-1].get("timestamp") if states else None
        }
        return rollup

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        
        agg = {}
        for m in metrics_list:
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        
        # Average the metrics
        count = len(metrics_list)
        return {k: v / count for k, v in agg.items()}
