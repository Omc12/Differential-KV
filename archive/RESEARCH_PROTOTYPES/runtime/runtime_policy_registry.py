from typing import Dict, Any, Callable

class RuntimePolicyRegistry:
    """
    Registry for reversible runtime policies.
    Policies control KV density, retrieval quality, bandwidth pressure, etc.
    """
    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {
            "kv_density": {"value": 0.5, "min": 0.1, "max": 1.0},
            "retrieval_k": {"value": 10, "min": 1, "max": 100},
            "bandwidth_limit": {"value": 1024, "min": 256, "max": 8192}, # MB/s
            "sparse_threshold": {"value": 0.05, "min": 0.01, "max": 0.5}
        }
        self.history = []

    def update_policy(self, name: str, new_value: Any, reason: str):
        """Updates a policy value with validation and logging."""
        if name not in self.policies:
            raise KeyError(f"Policy {name} not found.")
        
        p = self.policies[name]
        if isinstance(new_value, (int, float)):
            if new_value < p["min"] or new_value > p["max"]:
                raise ValueError(f"Value {new_value} out of range for {name} ({p['min']}-{p['max']})")
        
        old_value = p["value"]
        p["value"] = new_value
        self.history.append({
            "policy": name,
            "old": old_value,
            "new": new_value,
            "reason": reason
        })

    def get_policy(self, name: str) -> Any:
        return self.policies[name]["value"]

    def rollback(self):
        """Rolls back the last policy update."""
        if not self.history:
            return
        
        last = self.history.pop()
        self.policies[last["policy"]]["value"] = last["old"]
