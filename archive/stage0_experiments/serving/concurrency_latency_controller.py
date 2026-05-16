import time

class ConcurrencyLatencyController:
    """
    PHASE 11D: REAL CONCURRENCY & SERVING OPTIMIZATION
    
    Monitors and controls latency spikes under heavy concurrent load.
    Implements admission control to maintain target SLOs.
    """
    def __init__(self, target_latency: float = 0.1):
        self.target_latency = target_latency
        self.latency_history = []

    def record_latency(self, latency: float):
        self.latency_history.append(latency)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)

    def should_admit(self) -> bool:
        """
        Decision logic for admission control.
        """
        if not self.latency_history:
            return True
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        return avg_latency < self.target_latency * 1.5
