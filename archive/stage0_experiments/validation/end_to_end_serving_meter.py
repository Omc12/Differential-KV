import time

class ServingMeter:
    """
    Measures end-to-end serving performance.
    Calculates User-Visible Latency and Aggregate Throughput.
    """
    def __init__(self):
        self.completions = []
        self.start_time = time.perf_counter()

    def record_completion(self, request_id, token_count, total_latency):
        self.completions.append({
            "request_id": request_id,
            "tokens": token_count,
            "latency": total_latency,
            "timestamp": time.perf_counter()
        })

    def get_aggregate_metrics(self):
        if not self.completions:
            return {}
            
        total_tokens = sum(c['tokens'] for c in self.completions)
        total_time = time.perf_counter() - self.start_time
        avg_latency = sum(c['latency'] for c in self.completions) / len(self.completions)
        
        return {
            "total_requests": len(self.completions),
            "total_generated_tokens": total_tokens,
            "aggregate_tps": total_tokens / total_time if total_time > 0 else 0,
            "avg_user_visible_latency": avg_latency
        }
