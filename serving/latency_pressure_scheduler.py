from typing import List, Dict
import time

class LatencyPressureScheduler:
    """
    Schedules requests by balancing queue pressure and latency deadlines.
    Prevents starvation of long-running sparse queries.
    """
    def __init__(self):
        pass

    def rank_requests(self, requests: List[dict]) -> List[dict]:
        """
        Ranks requests by urgency and locality affinity.
        Urgency = (current_time - arrival_time) / deadline
        """
        now = time.perf_counter()
        
        def score(req):
            arrival = req.get("arrival_time", now)
            deadline = req.get("deadline", 1.0)
            urgency = (now - arrival) / deadline
            
            # Bonus for locality (if zone is already 'hot')
            # (Simplified for now)
            return urgency

        return sorted(requests, key=score, reverse=True)
