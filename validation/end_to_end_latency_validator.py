"""
validation/end_to_end_latency_validator.py

Validates the full-stack latency of retrieval and attention.
Separates 'time-to-first-token' (TTFT) from 'inter-token latency' (ITL).
"""

import numpy as np
from typing import List, Dict, Any
import logging

class EndToEndLatencyValidator:
    """
    Validates user-visible latency metrics.
    """
    def __init__(self):
        self.logger = logging.getLogger("EndToEndLatencyValidator")

    def calculate_p99_latency(self, latency_series: List[float]) -> float:
        """Calculates p99 tail latency."""
        if not latency_series: return 0.0
        return np.percentile(latency_series, 99)

    def validate_latency_budget(self, measured: float, budget: float) -> bool:
        """Checks if measured latency exceeds the architectural budget."""
        if measured > budget:
            self.logger.warning(f"LATENCY BUDGET EXCEEDED: {measured:.2f}ms > {budget:.2f}ms")
            return False
        return True

    def analyze_jitter(self, timestamps: List[float]) -> float:
        """Measures arrival jitter to detect synchronization stalls."""
        if len(timestamps) < 2: return 0.0
        intervals = np.diff(timestamps)
        return np.std(intervals)
