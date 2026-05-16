from typing import Dict, List, Any

class TelemetryScopeTracker:
    """
    Tracks which GPU allocations, kernels, and runtimes were measured.
    """
    def __init__(self):
        self.scope = {
            "gpu_allocations": False,
            "kernels": False,
            "runtimes": False,
            "flops": False,
            "model_weights": False,
            "wall_clock": False
        }

    def set_scope(self, item: str, included: bool = True):
        if item in self.scope:
            self.scope[item] = included

    def get_scope_manifest(self) -> Dict[str, bool]:
        return self.scope

# Global singleton
scope_tracker = TelemetryScopeTracker()
