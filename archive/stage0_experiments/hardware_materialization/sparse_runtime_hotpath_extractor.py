"""
hardware_materialization/sparse_runtime_hotpath_extractor.py

Identifies runtime bottlenecks and hotpaths in sparse execution.
"""

import time
from collections import defaultdict
from typing import Dict, List, Any

class SparseRuntimeHotpathExtractor:
    """
    Traces and ranks runtime stages based on execution frequency and duration.
    """
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.launch_counts = defaultdict(int)

    def trace_stage(self, stage_name: str, duration_ms: float):
        """Records a single execution of a stage."""
        self.execution_times[stage_name].append(duration_ms)
        self.launch_counts[stage_name] += 1

    def get_bottlenecks(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Analyzes recorded data and returns top bottlenecks."""
        rankings = []
        for name, times in self.execution_times.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            count = self.launch_counts[name]
            rankings.append({
                "stage": name,
                "avg_ms": avg_time,
                "total_ms": total_time,
                "launches": count
            })
        
        # Sort by total time descending
        return sorted(rankings, key=lambda x: x["total_ms"], reverse=True)[:top_k]

    def reset(self):
        self.execution_times.clear()
        self.launch_counts.clear()

    def report(self) -> str:
        """Generates a human-readable hotpath report."""
        bottlenecks = self.get_bottlenecks()
        if not bottlenecks:
            return "No hotpath data collected."
            
        lines = ["--- Runtime Hotpath Analysis ---"]
        for b in bottlenecks:
            lines.append(f"Stage: {b['stage']:<25} | Total: {b['total_ms']:8.2f}ms | Avg: {b['avg_ms']:6.2f}ms | Launches: {b['launches']}")
        return "\n".join(lines)
