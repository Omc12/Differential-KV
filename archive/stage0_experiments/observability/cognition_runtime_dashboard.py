import json
from typing import Dict

class CognitionRuntimeDashboard:
    """
    Generates a real-time web-compatible data feed (JSON) for monitoring
    the health, entropy, and manifold states of a cognition cluster.
    """
    def __init__(self):
        self.metrics = {}

    def update_metrics(self, active_nodes: int, global_entropy: float, exchange_rate: float):
        self.metrics = {
            "active_nodes": active_nodes,
            "global_entropy": round(global_entropy, 4),
            "manifold_exchange_rate_mbps": round(exchange_rate, 2),
            "status": "stable" if global_entropy < 0.6 else "warning"
        }

    def render_feed(self) -> str:
        return json.dumps(self.metrics, indent=2)

if __name__ == "__main__":
    dash = CognitionRuntimeDashboard()
    dash.update_metrics(12, 0.45, 120.5)
    print(dash.render_feed())
