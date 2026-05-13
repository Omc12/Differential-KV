"""
validation/telemetry_consistency_checker.py

Cross-checks metrics from multiple sources (Host, GPU, Network).
Ensures that data sent over network matches data received by host.
"""

from typing import Dict, Any
import logging

class TelemetryConsistencyChecker:
    """
    Consistency auditor for distributed telemetry.
    """
    def __init__(self):
        self.logger = logging.getLogger("TelemetryConsistencyChecker")

    def verify_network_traffic(self, sent_bytes: int, received_bytes: int, loss_tolerance: float = 0.01) -> bool:
        """
        Ensures distributed communication stats match across nodes.
        """
        if sent_bytes == 0 and received_bytes == 0:
            return True
            
        ratio = abs(sent_bytes - received_bytes) / max(sent_bytes, received_bytes)
        is_consistent = ratio < loss_tolerance
        
        if not is_consistent:
            self.logger.error(f"TELEMETRY INCONSISTENCY: Network Sent ({sent_bytes}) != Received ({received_bytes})")
            
        return is_consistent

    def verify_memory_usage(self, reported_allocated: int, hw_reserved: int) -> bool:
        """
        Checks if hardware reserved memory covers reported allocations.
        """
        return hw_reserved >= reported_allocated
