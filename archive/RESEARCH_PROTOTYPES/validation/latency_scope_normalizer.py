class LatencyScopeNormalizer:
    """
    Normalizes latency measurements across different scopes (prefill, decode, serving).
    Ensures that "latency" means the same thing in all reports.
    """
    def __init__(self):
        pass

    def normalize(self, metrics: dict):
        # Ensure 'latency' is always total end-to-end
        # Ensure 'per_token_latency' is specifically for decode
        total = metrics.get("total_time", metrics.get("latency", 0))
        tokens = metrics.get("output_len", metrics.get("token_count", 1))
        
        return {
            "e2e_latency": total,
            "decode_only_latency": metrics.get("decode_time", total),
            "ms_per_token": (metrics.get("decode_time", total) / tokens) * 1000 if tokens > 0 else 0
        }
