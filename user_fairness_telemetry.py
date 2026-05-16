import logging
from typing import Dict, List, Any
import numpy as np

class UserFairnessTelemetry:
    """
    Measures per-user TPS, queue wait variance, and fairness index.
    """
    def __init__(self):
        self.logger = logging.getLogger("UserFairnessTelemetry")
        self.user_data = {} # session_id -> {tps: [], wait_times: []}

    def record_user_request(self, session_id: str, tps: float, wait_time_ms: float):
        if session_id not in self.user_data:
            self.user_data[session_id] = {"tps": [], "wait_times": []}
        
        self.user_data[session_id]["tps"].append(tps)
        self.user_data[session_id]["wait_times"].append(wait_time_ms)

    def get_fairness_report(self) -> Dict[str, Any]:
        """
        Calculates Jain's Fairness Index for per-user throughput.
        """
        if not self.user_data:
            return {"fairness_index": 1.0, "skew": 0}
            
        avg_tps_per_user = [np.mean(d["tps"]) for d in self.user_data.values() if d["tps"]]
        if not avg_tps_per_user:
            return {"fairness_index": 1.0, "skew": 0}
            
        n = len(avg_tps_per_user)
        sum_x = sum(avg_tps_per_user)
        sum_x2 = sum(x**2 for x in avg_tps_per_user)
        
        # Jain's Fairness Index = (sum x)^2 / (n * sum x^2)
        fairness_index = (sum_x**2) / (n * sum_x2) if sum_x2 > 0 else 1.0
        
        # Skew: ratio of max tps to min tps
        skew = max(avg_tps_per_user) / min(avg_tps_per_user) if min(avg_tps_per_user) > 0 else 0
        
        return {
            "fairness_index": float(fairness_index),
            "tps_skew": float(skew),
            "user_count": n,
            "worst_user_tps": float(min(avg_tps_per_user)),
            "best_user_tps": float(max(avg_tps_per_user))
        }

    def clear(self):
        self.user_data = {}
