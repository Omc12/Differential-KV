import logging
import json
import time
from typing import Dict, Any, List

class TelemetryConsolidationLayer:
    """
    Unifies disparate telemetry systems into a coherent observability stack.
    Consolidates: benchmark, serving, latency, residency, integrity, and portability telemetry.
    """
    def __init__(self):
        self.logger = logging.getLogger("TelemetryConsolidationLayer")
        # Global metric store
        self.unified_metrics = {
            "serving": {},
            "latency": {},
            "residency": {},
            "integrity": {},
            "portability": {},
            "operational": {}
        }

    def ingest_metrics(self, category: str, metrics: Dict[str, Any]):
        """Ingests metrics from a specific subsystem."""
        if category not in self.unified_metrics:
            self.unified_metrics[category] = {}
        self.unified_metrics[category].update(metrics)
        self.unified_metrics["operational"]["last_sync"] = time.time()

    def get_unified_snapshot(self) -> Dict[str, Any]:
        """Returns a single coherent observability snapshot."""
        return self.unified_metrics

    def export_telemetry(self, path: str = "./unified_telemetry.json"):
        """Persists the unified telemetry to disk."""
        with open(path, 'w') as f:
            json.dump(self.unified_metrics, f, indent=4)
        self.logger.info(f"Unified telemetry exported to {path}")

    def consolidate_legacy_logs(self, log_files: List[str]):
        """Migrates legacy log data into the unified stack (Optional)."""
        pass
