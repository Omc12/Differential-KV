"""
hardware_materialization/kernel_bottleneck_analyzer.py

Identifies actual GPU bottlenecks including hotspots, launch overhead, and sync stalls.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("BottleneckAnalyzer")

class KernelBottleneckAnalyzer:
    """
    Analyzes recorded trace data to rank bottlenecks by impact.
    """
    def __init__(self):
        self.bottlenecks: List[Dict[str, Any]] = []

    def analyze_traces(self, trace_summary: Dict[str, float]):
        """
        Ranks kernels and operations by their contribution to latency.
        """
        self.bottlenecks = []
        # Sort by duration descending
        sorted_stages = sorted(trace_summary.items(), key=lambda x: x[1], reverse=True)
        
        for name, avg_ms in sorted_stages:
            impact = "HIGH" if avg_ms > 1.0 else "MEDIUM" if avg_ms > 0.1 else "LOW"
            self.bottlenecks.append({
                "stage": name,
                "avg_ms": avg_ms,
                "impact": impact,
                "category": self._categorize_stage(name)
            })
            
        return self.bottlenecks

    def _categorize_stage(self, name: str) -> str:
        if "sync" in name.lower() or "wait" in name.lower():
            return "Synchronization"
        if "triton" in name.lower() or "kernel" in name.lower():
            return "Kernel"
        if "replay" in name.lower() or "graph" in name.lower():
            return "Overhead"
        return "Generic"

    def get_hottest_kernel(self) -> Optional[str]:
        if not self.bottlenecks:
            return None
        return self.bottlenecks[0]["stage"]

    def report(self) -> str:
        if not self.bottlenecks:
            return "No bottleneck analysis available."
            
        lines = ["--- GPU Bottleneck Report ---"]
        for b in self.bottlenecks:
            lines.append(f"[{b['impact']:>6}] {b['stage']:<25} | {b['avg_ms']:8.4f}ms | {b['category']}")
        return "\n".join(lines)
