"""
validation/distributed_truth_reconciler.py

Reconciles performance metrics across multiple distributed nodes.
Ensures that sum(node_tps) matches aggregate_tps and detects 'double counting'.
"""

from typing import Dict, List, Any
import logging

class DistributedTruthReconciler:
    """
    Consistency auditor for cluster-wide metrics.
    """
    def __init__(self):
        self.logger = logging.getLogger("DistributedTruthReconciler")

    def reconcile_tps(self, node_metrics: Dict[int, float], aggregate_tps: float) -> bool:
        """
        Verifies that the sum of individual node TPS matches the reported aggregate.
        Prevents 'double counting' of tokens in distributed runs.
        """
        node_sum = sum(node_metrics.values())
        diff = abs(node_sum - aggregate_tps)
        
        if diff / aggregate_tps > 0.01:
            self.logger.error(f"TPS RECONCILIATION FAILED: Node Sum ({node_sum}) != Aggregate ({aggregate_tps})")
            return False
            
        return True

    def check_clock_drift(self, node_timestamps: List[float]) -> float:
        """
        Checks for clock drift between nodes which might corrupt latency metrics.
        """
        # Simplistic drift check
        return max(node_timestamps) - min(node_timestamps)
