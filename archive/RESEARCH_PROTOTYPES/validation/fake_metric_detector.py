"""
validation/fake_metric_detector.py

Heuristics and statistical checks to detect synthetic or placeholder data.
Rejects metrics with zero variance or impossible hardware-to-software correlations.
"""

import numpy as np
from typing import List, Dict
import logging

class FakeMetricDetector:
    """
    Statistical adversary that looks for 'faked' performance data.
    """
    def __init__(self):
        self.logger = logging.getLogger("FakeMetricDetector")

    def detect_low_variance(self, metric_series: List[float], threshold: float = 1e-6) -> bool:
        """
        Detects metrics that have suspiciously low variance (likely mocked).
        """
        if len(metric_series) < 10:
            return False
            
        std_dev = np.std(metric_series)
        is_synthetic = std_dev < threshold
        
        if is_synthetic:
            self.logger.error(f"SYNTHETIC METRIC DETECTED: Variance too low ({std_dev:.8f})")
            
        return is_synthetic

    def detect_impossible_occupancy(self, reported_occupancy: float, reported_tps: float) -> bool:
        """
        Checks for impossible combinations of occupancy and throughput.
        E.g., 99% occupancy with only 10 TPS.
        """
        if reported_occupancy > 0.9 and reported_tps < 10:
            self.logger.error(f"IMPOSSIBLE METRIC DETECTED: High occupancy ({reported_occupancy}) with low TPS ({reported_tps})")
            return True
        return False

    def verify_concurrency(self, process_timestamps: List[List[float]]) -> bool:
        """
        Verifies that multiple processes were actually executing concurrently 
        by checking overlapping timestamp ranges.
        """
        if len(process_timestamps) < 2:
            return False # Not distributed
            
        ranges = [(min(pts), max(pts)) for pts in process_timestamps]
        
        # Check for overlap
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                start1, end1 = ranges[i]
                start2, end2 = ranges[j]
                
                if max(start1, start2) < min(end1, end2):
                    return True # Real overlap detected
                    
        self.logger.error("FAKE CONCURRENCY: Sequential execution detected in distributed run logs.")
        return False
