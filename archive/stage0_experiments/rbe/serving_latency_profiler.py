
import time
from typing import Dict, List, Any

class ServingLatencyProfiler:
    """
    PHASE 24.2: Serving Latency Profiler (RBE).
    Measures TTFT, ITL, and total generation latency.
    """
    def __init__(self):
        self.request_timings = {}
        
    def start_request(self, request_id: str):
        self.request_timings[request_id] = {
            "start_time": time.perf_counter(),
            "first_token_time": None,
            "token_times": []
        }
        
    def record_token(self, request_id: str):
        if request_id in self.request_timings:
            now = time.perf_counter()
            if self.request_timings[request_id]["first_token_time"] is None:
                self.request_timings[request_id]["first_token_time"] = now
            self.request_timings[request_id]["token_times"].append(now)
            
    def get_metrics(self, request_id: str) -> Dict[str, float]:
        if request_id not in self.request_timings:
            return {}
            
        t = self.request_timings[request_id]
        ttft = (t["first_token_time"] - t["start_time"]) * 1000 if t["first_token_time"] else 0.0
        
        itls = []
        if len(t["token_times"]) > 1:
            for i in range(1, len(t["token_times"])):
                itls.append((t["token_times"][i] - t["token_times"][i-1]) * 1000)
                
        avg_itl = sum(itls) / len(itls) if itls else 0.0
        total_time = (t["token_times"][-1] - t["start_time"]) if t["token_times"] else 0.0
        tps = len(t["token_times"]) / total_time if total_time > 0 else 0.0
        
        return {
            "ttft_ms": ttft,
            "avg_itl_ms": avg_itl,
            "total_latency_s": total_time,
            "tps": tps
        }
