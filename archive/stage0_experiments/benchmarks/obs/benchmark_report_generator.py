"""
benchmarks/obs/benchmark_report_generator.py

Benchmark report generator for Differential KV.
Creates Markdown and JSON reports for publication.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional

class BenchmarkReportGenerator:
    """
    Generates structured reports from benchmark results.
    """
    def __init__(self, output_dir: str = "results/obs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_markdown_report(self, metrics: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> str:
        """Creates a human-readable Markdown summary."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        report = [
            f"# Differential KV — Operational Benchmark Report",
            f"Generated at: {timestamp}",
            "",
            "## 1. Key Performance Indicators",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Sustained Sparse TPS | {metrics.get('sustained_sparse_tps', 0):.2f} |",
            f"| TTFT (ms) | {metrics.get('ttft_ms', 0):.2f} |",
            f"| ITL (ms) | {metrics.get('itl_ms', 0):.2f} |",
            f"| VRAM Efficiency Ratio | {metrics.get('vram_efficiency_ratio', 0):.2f}x |",
            f"| Long-Context Scaling | {metrics.get('long_context_scaling_factor', 0):.4f} |",
            "",
            "## 2. Integrity & Stability",
            f"- Benchmark Reproducibility: {metrics.get('benchmark_reproducibility', 0):.2%}",
            f"- Serving Stability Index: {metrics.get('serving_stability_index', 0):.4f}",
            f"- Replay Consistency Score: {metrics.get('replay_consistency_score', 0):.2%}",
            ""
        ]
        
        if comparison:
            report.append("## 3. Competitive Runtime Comparison")
            report.append("| Runtime | TPS | VRAM (GB) | Relative Gain |")
            report.append("| :--- | :--- | :--- | :--- |")
            base_tps = comparison.get("transformers", {}).get("tps", 1)
            for name, data in comparison.items():
                gain = data['tps'] / base_tps
                report.append(f"| {name} | {data['tps']:.1f} | {data['vram_usage_gb']:.1f} | {gain:.2f}x |")
            report.append("")

        report_content = "\n".join(report)
        
        filepath = os.path.join(self.output_dir, f"report_{int(time.time())}.md")
        with open(filepath, "w") as f:
            f.write(report_content)
        
        return filepath

    def export_json(self, data: Dict[str, Any]) -> str:
        """Exports full telemetry to JSON."""
        filepath = os.path.join(self.output_dir, f"telemetry_{int(time.time())}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        return filepath

if __name__ == "__main__":
    generator = BenchmarkReportGenerator()
    print("BenchmarkReportGenerator module loaded.")
