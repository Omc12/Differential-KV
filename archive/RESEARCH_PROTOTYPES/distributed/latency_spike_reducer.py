"""
distributed/latency_spike_reducer.py

Minimizes tail latency spikes in distributed runs.
Detects and bypasses 'congested' nodes during high-pressure bursts.
"""

from typing import List, Dict, Any
import time
import logging

class LatencySpikeReducer:
    """
    Monitor and mitigation system for tail latency.
    """
    def __init__(self, spike_threshold_ms: float = 200.0):
        self.threshold = spike_threshold_ms
        self.node_latencies: Dict[int, List[float]] = {}
        self.logger = logging.getLogger("LatencySpikeReducer")

    def record_latency(self, node_id: int, latency_ms: float):
        """Records latency and checks for spikes."""
        if node_id not in self.node_latencies:
            self.node_latencies[node_id] = []
        
        self.node_latencies[node_id].append(latency_ms)
        if len(self.node_latencies[node_id]) > 100:
            self.node_latencies[node_id].pop(0)
            
        if latency_ms > self.threshold:
            self.logger.warning(f"LATENCY SPIKE on Node {node_id}: {latency_ms:.2f}ms")

    def is_congested(self, node_id: int) -> bool:
        """Determines if a node is currently experiencing a spike."""
        if node_id not in self.node_latencies or not self.node_latencies[node_id]:
            return False
        
        avg = sum(self.node_latencies[node_id]) / len(self.node_latencies[node_id])
        return self.node_latencies[node_id][-1] > avg * 2.0
