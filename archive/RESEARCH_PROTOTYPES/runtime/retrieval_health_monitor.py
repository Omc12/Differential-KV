import torch

class RetrievalHealthMonitor:
    """
    Tracks long-term retrieval health across large contexts.
    Provides telemetry for sparse scaling stability.
    """
    def __init__(self):
        self.health_history = []

    def update(self, needle_retrieval_success: bool, latency: float):
        self.health_history.append({
            "success": needle_retrieval_success,
            "latency": latency,
            "timestamp": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        })

    def get_summary(self):
        if not self.health_history:
            return "No data"
            
        success_rate = sum(h["success"] for h in self.health_history) / len(self.health_history)
        avg_latency = sum(h["latency"] for h in self.health_history) / len(self.health_history)
        
        return {
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_samples": len(self.health_history)
        }
