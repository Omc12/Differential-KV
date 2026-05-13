from runtime.runtime_policy_registry import RuntimePolicyRegistry
from typing import Dict, Any

class AdaptivePolicyOptimizer:
    """
    Optimizes runtime policies based on explicit reward targets (latency, efficiency).
    Uses a bounded optimization search.
    """
    def __init__(self, registry: RuntimePolicyRegistry):
        self.registry = registry

    def optimize_step(self, metrics: Dict[str, float]):
        """
        Adjusts policies based on current hardware metrics.
        metrics: {'latency': ms, 'memory_usage': %, 'retrieval_accuracy': 0-1}
        """
        # Example logic: if latency is high, reduce KV density
        if metrics.get("latency", 0) > 100: # Threshold 100ms
            current_density = self.registry.get_policy("kv_density")
            self.registry.update_policy("kv_density", max(0.1, current_density - 0.05), "Latency reduction")
            
        # If memory is low, increase KV density for better quality
        if metrics.get("memory_usage", 100) < 50:
            current_density = self.registry.get_policy("kv_density")
            self.registry.update_policy("kv_density", min(1.0, current_density + 0.05), "Quality optimization")

        # If retrieval accuracy is low, increase retrieval_k
        if metrics.get("retrieval_accuracy", 1.0) < 0.8:
            current_k = self.registry.get_policy("retrieval_k")
            self.registry.update_policy("retrieval_k", min(100, current_k + 2), "Accuracy recovery")
