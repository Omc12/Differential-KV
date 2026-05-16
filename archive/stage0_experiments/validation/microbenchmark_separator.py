"""
validation/microbenchmark_separator.py

Separates internal microbenchmarks from end-to-end serving performance.
Prevents 'fused kernel throughput' from being presented as 'serving throughput'.
"""

from typing import Dict, Any, List
import logging

class MicrobenchmarkSeparator:
    """
    Splits a results report into 'Micro' (internal) and 'Macro' (user-facing) tiers.
    """
    def __init__(self):
        self.logger = logging.getLogger("MicrobenchmarkSeparator")

    def tier_results(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Groups results by their granularity.
        """
        tiered = {
            "macro_serving": {},
            "micro_kernel": {},
            "retrieval_ops": {},
            "infrastructure": {}
        }
        
        for name, value in results.items():
            name_lower = name.lower()
            if "tps" in name_lower and "serving" in name_lower:
                tiered["macro_serving"][name] = value
            elif "kernel" in name_lower or "fused" in name_lower:
                tiered["micro_kernel"][name] = value
            elif "retrieval" in name_lower or "anchor" in name_lower:
                tiered["retrieval_ops"][name] = value
            else:
                tiered["infrastructure"][name] = value
                
        return tiered

    def verify_no_label_bleed(self, tiered_results: Dict[str, Any]) -> bool:
        """
        Ensures microbenchmark metrics haven't 'bled' into macro reporting.
        """
        macro = tiered_results.get("macro_serving", {})
        for name in macro:
            if "kernel" in name.lower() or "flops" in name.lower():
                self.logger.error(f"LABEL BLEED: Micro-metric '{name}' found in Macro Serving group!")
                return False
        return True
