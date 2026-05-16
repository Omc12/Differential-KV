"""
validation/distributed_metric_consistency.py

Checks consistency of metrics in multi-node runs.
Detects impossible scaling jumps and ensures conservation of throughput.
"""

from typing import Dict, List, Any
import logging

class DistributedMetricConsistency:
    """
    Consistency auditor for distributed benchmarks.
    """
    def __init__(self):
        self.logger = logging.getLogger("DistributedMetricConsistency")

    def check_scaling_jumps(self, node_count_series: List[int], tps_series: List[float]) -> bool:
        """
        Detects impossible super-linear scaling (> 1.2x per node).
        """
        if len(node_count_series) < 2: return True
        
        for i in range(1, len(node_count_series)):
            node_ratio = node_count_series[i] / node_count_series[i-1]
            tps_ratio = tps_series[i] / tps_series[i-1]
            
            # Allow for some measurement noise (up to 20% over ideal)
            if tps_ratio > node_ratio * 1.2:
                self.logger.error(f"IMPOSSIBLE SCALING: Nodes {node_count_series[i-1]}->{node_count_series[i]} resulted in {tps_ratio:.2f}x speedup.")
                return False
        return True

    def verify_request_conservation(self, node_requests: List[int], aggregate_requests: int) -> bool:
        """
        Ensures that sum of requests across nodes matches aggregate.
        """
        node_sum = sum(node_requests)
        if node_sum != aggregate_requests:
            self.logger.error(f"REQUEST CONSERVATION FAILED: Node Sum ({node_sum}) != Aggregate ({aggregate_requests})")
            return False
        return True
