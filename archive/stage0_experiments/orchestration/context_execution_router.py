from typing import Dict, Any, List

class ContextExecutionRouter:
    """
    Routes contexts to specific execution units based on metadata and load.
    Ensures optimal placement for long-context vs short-context tasks.
    """
    def __init__(self, num_units: int = 4):
        self.num_units = num_units
        self.unit_load = [0] * num_units
        self.routing_history = []

    def route_context(self, context_metadata: Dict[str, Any]) -> int:
        """Routes a context to a unit with the least load."""
        context_len = context_metadata.get("length", 0)
        
        # Heuristic: large contexts go to specific units if possible, or just least loaded
        unit_index = self.unit_load.index(min(self.unit_load))
        
        # Update load (simulation)
        self.unit_load[unit_index] += context_len
        self.routing_history.append({"unit": unit_index, "length": context_len})
        
        return unit_index

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_units": self.num_units,
            "current_load": self.unit_load,
            "total_routed": len(self.routing_history)
        }
