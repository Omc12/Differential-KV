import time
import collections

class TokenTracker:
    """
    Tracks generated tokens in real-time.
    Distinguishes between internal overhead and true generation throughput.
    """
    def __init__(self):
        self.request_token_times = collections.defaultdict(list)
        self.request_start_times = {}

    def record_token(self, request_id, token, latency):
        """
        Records a single generated token and its per-token latency.
        """
        self.request_token_times[request_id].append({
            "token": token,
            "latency": latency,
            "timestamp": time.perf_counter()
        })

    def get_stats(self, request_id):
        if request_id not in self.request_token_times:
            return None
            
        times = [t['latency'] for t in self.request_token_times[request_id]]
        if not times:
            return None
            
        avg_latency = sum(times) / len(times)
        p50 = sorted(times)[len(times)//2]
        p99 = sorted(times)[int(len(times)*0.99)]
        
        return {
            "token_count": len(times),
            "avg_token_latency": avg_latency,
            "p50_latency": p50,
            "p99_latency": p99,
            "tokens_per_sec": 1.0 / avg_latency if avg_latency > 0 else 0
        }
