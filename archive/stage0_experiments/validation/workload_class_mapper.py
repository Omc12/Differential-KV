"""
validation/workload_class_mapper.py

Explicitly maps benchmarks to workload types (Inference, Retrieval, Sync).
Ensures that metrics are grouped by their physical execution class.
"""

from typing import Dict, List, Any
import logging

class WorkloadClassMapper:
    """
    Groups metrics by their operational workload.
    """
    CLASSES = ["INFERENCE", "RETRIEVAL", "SYNCHRONIZATION", "INFRASTRUCTURE"]

    def __init__(self):
        self.logger = logging.getLogger("WorkloadClassMapper")

    def map_to_class(self, metric_name: str) -> str:
        """
        Determines the operational class for a given metric.
        """
        name = metric_name.upper()
        if any(x in name for x in ["TPS", "GENERATE", "ATTENTION"]):
            return "INFERENCE"
        if any(x in name for x in ["RETRIEVAL", "SHARD", "ANCHOR"]):
            return "RETRIEVAL"
        if any(x in name for x in ["SYNC", "COMM", "ALL_REDUCE"]):
            return "SYNCHRONIZATION"
        return "INFRASTRUCTURE"

    def group_report(self, report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Groups a report into workload-specific sections.
        """
        grouped = {c: {} for c in self.CLASSES}
        for name, value in report.items():
            cls = self.map_to_class(name)
            grouped[cls][name] = value
        return grouped
