"""
validation/serving_vs_kernel_splitter.py

Strict separation logic for serving vs kernel throughput.
Prevents internal 'kernel ops' from being presented as 'serving capacity'.
"""

from typing import Dict, Any, Tuple
import logging

class ServingVsKernelSplitter:
    """
    Enforces a wall between macro-serving and micro-kernel metrics.
    """
    def __init__(self):
        self.logger = logging.getLogger("ServingVsKernelSplitter")

    def split_report(self, report: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Splits report into (serving_metrics, kernel_metrics).
        """
        serving = {}
        kernel = {}
        
        for name, value in report.items():
            if "serving" in name.lower() or "tps" in name.lower():
                serving[name] = value
            elif "kernel" in name.lower() or "fused" in name.lower() or "flops" in name.lower():
                kernel[name] = value
            else:
                # Default to kernel if ambiguous
                kernel[name] = value
                
        return serving, kernel

    def verify_no_contamination(self, serving: Dict[str, Any], kernel: Dict[str, Any]) -> bool:
        """
        Verifies that no kernel-level metrics are in the serving group.
        """
        for name in serving:
            if "fused" in name.lower() or "triton" in name.lower():
                self.logger.error(f"CONTAMINATION DETECTED: Kernel metric '{name}' in Serving group.")
                return False
        return True
