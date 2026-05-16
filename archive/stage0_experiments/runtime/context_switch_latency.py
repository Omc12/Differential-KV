import time

class ContextSwitchLatency:
    """
    Measures the technical cost (latency) of switching between 
    large files or disparate contexts.
    """
    def __init__(self):
        self.switches = []

    def record_switch(self, from_ctx: str, to_ctx: str, duration: float):
        self.switches.append({
            "from": from_ctx,
            "to": to_ctx,
            "latency": duration
        })

    def get_p99_latency(self):
        if not self.switches: return 0.0
        latencies = sorted([s['latency'] for s in self.switches])
        return latencies[int(len(latencies) * 0.99)]
