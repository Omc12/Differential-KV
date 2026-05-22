"""
validation/metric_semantic_reconciler.py

Reconciles historical and current metrics to ensure semantic consistency.
Eliminates ambiguity by explicitly labeling every throughput metric.
"""

from typing import Dict, Any, List
from validation.metric_taxonomy import MetricClass
import logging

class MetricSemanticReconciler:
    """
    Enforces semantic correctness on performance reports.
    """
    def __init__(self):
        self.logger = logging.getLogger("MetricSemanticReconciler")

    def reconcile_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a report and adds semantic metadata to every metric.
        """
        reconciled = {}
        for name, value in report.items():
            metadata = self._get_semantic_metadata(name, value)
            reconciled[name] = {
                "value": value,
                "semantics": metadata
            }
        return reconciled

    def _get_semantic_metadata(self, name: str, value: Any) -> Dict[str, str]:
        """
        Extracts semantic context for a metric.
        """
        name_lower = name.lower()
        
        metadata = {
            "workload_type": "UNKNOWN",
            "token_definition": "RAW_TOKENS",
            "execution_scope": "SINGLE_NODE",
            "hardware_scope": "GPU",
            "verifiability": "UNVERIFIED"
        }
        
        if "serving" in name_lower or "tps" in name_lower:
            metadata["workload_type"] = "INFERENCE_SERVING"
            metadata["token_definition"] = "GENERATED_TOKENS"
        elif "kernel" in name_lower:
            metadata["workload_type"] = "KERNEL_MICROBENCHMARK"
            metadata["token_definition"] = "THROUGHPUT_TOKENS"
        elif "retrieval" in name_lower:
            metadata["workload_type"] = "SPARSE_RETRIEVAL"
            metadata["token_definition"] = "RETRIEVED_BLOCKS"
            
        if "distributed" in name_lower or "multi" in name_lower:
            metadata["execution_scope"] = "DISTRIBUTED_CLUSTER"
            
        return metadata
