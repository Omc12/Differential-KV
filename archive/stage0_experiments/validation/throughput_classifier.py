"""
validation/throughput_classifier.py

Classifies incoming performance metrics into correct taxonomy categories.
Prevents internal operations/sec from being labeled as end-to-end serving TPS.
"""

from typing import Dict, Any, List
from validation.metric_taxonomy import MetricClass, TAXONOMY_MAP
import logging

class ThroughputClassifier:
    """
    Adversarial classifier that re-labels metrics to prevent inflation.
    """
    def __init__(self):
        self.logger = logging.getLogger("ThroughputClassifier")

    def classify_metric(self, name: str, value: Any) -> MetricClass:
        """
        Determines the correct scientific class for a given metric name.
        """
        name_lower = name.lower()
        
        # 1. Direct Mapping
        for key, mclass in TAXONOMY_MAP.items():
            if key in name_lower:
                return mclass
                
        # 2. Heuristic Classification
        if "ops" in name_lower or "queries" in name_lower:
            return MetricClass.RETRIEVAL_OPS
        if "kernel" in name_lower:
            return MetricClass.KERNEL_THROUGHPUT
        if "sim" in name_lower:
            return MetricClass.SIMULATED
        if "est" in name_lower:
            return MetricClass.THEORETICAL
            
        return MetricClass.MICROBENCHMARK

    def sanitize_report(self, raw_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Transforms a raw report into a taxonomy-compliant report.
        """
        sanitized = {}
        for name, value in raw_report.items():
            mclass = self.classify_metric(name, value)
            sanitized[name] = {
                "value": value,
                "class": mclass.value,
                "scientific_label": f"[{mclass.name}] {name}"
            }
            
            if mclass == MetricClass.SERVING_TPS and value > 10000:
                self.logger.warning(f"SUSPICIOUS TPS: {value} classified as {mclass.value}. Verify raw logs.")
                
        return sanitized
