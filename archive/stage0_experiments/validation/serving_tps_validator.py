"""
validation/serving_tps_validator.py

Specific validator for End-to-End Serving TPS.
Ensures that 'Serving TPS' includes full-stack overhead, not just kernel speed.
"""

import time
from typing import List, Dict, Any
import logging

class ServingTPSValidator:
    """
    Validates serving throughput by measuring actual user-visible token delivery.
    """
    def __init__(self):
        self.logger = logging.getLogger("ServingTPSValidator")

    def validate_tps_log(self, request_logs: List[Dict[str, Any]]) -> float:
        """
        Calculates REAL Serving TPS from a list of request completion events.
        tps = total_tokens / (last_completion - first_submission)
        """
        if not request_logs:
            return 0.0
            
        completions = [r['timestamp'] for r in request_logs if 'timestamp' in r]
        submissions = [r.get('start_time', r['timestamp']) for r in request_logs]
        tokens = [r.get('tokens', 0) for r in request_logs]
        
        if not completions:
            return 0.0
            
        total_time = max(completions) - min(submissions)
        total_tokens = sum(tokens)
        
        if total_time <= 0:
            return 0.0
            
        real_tps = total_tokens / total_time
        self.logger.info(f"Verified Serving TPS: {real_tps:.2f} (Total Time: {total_time:.2f}s)")
        
        return real_tps

    def detect_synthetic_tps(self, tps_series: List[float]) -> bool:
        """
        Detects if TPS reporting is synthetic (e.g. perfectly constant).
        """
        import numpy as np
        if len(tps_series) < 5: return False
        return np.std(tps_series) < 1e-9
