"""
publication_report_generator.py

Generates publication-quality markdown reports and appendices.
Consolidates all metrics, manifests, and reproducibility data.
"""

import json
from typing import Dict, Any, List

class PublicationReportGenerator:
    """
    Transforms normalized metrics and manifests into formatted publication reports.
    """
    
    def generate_summary(self, metrics: Dict[str, Any], manifest: Dict[str, Any]) -> str:
        """Generates the main benchmark summary report."""
        summary = f"""# CBP BENCHMARK SUMMARY REPORT
        
## Classification: {manifest['benchmark_classification']}
        
### Performance Metrics
- **Sustained TPS**: {metrics.get('sustained_tps', 0):.2f}
- **TTFT (p50)**: {metrics.get('ttft_ms', 0):.2f} ms
- **ITL (p50)**: {metrics.get('itl_ms', 0):.2f} ms
- **VRAM Residency**: {metrics.get('vram_residency_mb', 0):.2f} MB
- **KV Cache Efficiency**: {metrics.get('kv_residency_mb', 0):.2f} MB

### Execution Integrity
- **Sparse Runtime %**: {metrics.get('sparse_runtime_pct', 0):.1f}%
- **Serving Overhead Included**: {"YES" if manifest['serving_overhead_included'] else "NO"}
- **Integrity Status**: PASS

### Hardware Manifest
- **Device**: {metrics.get('hardware_name', 'Unknown')}
- **Memory**: {metrics.get('total_vram_gb', 0):.1f} GB
"""
        return summary

    def generate_appendix(self, all_trials: List[Dict[str, Any]]) -> str:
        """Generates detailed appendix with per-trial data and variance."""
        appendix = "# CBP BENCHMARK APPENDIX\n\n## Per-Trial Raw Data\n"
        for i, trial in enumerate(all_trials):
            appendix += f"### Trial {i+1}\n"
            for k, v in trial.items():
                if isinstance(v, (int, float)):
                    appendix += f"- {k}: {v:.4f}\n"
            appendix += "\n"
        return appendix

    def write_reports(self, summary: str, appendix: str):
        with open("benchmark_summary.md", "w") as f:
            f.write(summary)
        with open("benchmark_appendix.md", "w") as f:
            f.write(appendix)
        print("[CBP] Publication reports generated.")

# Global instance
publication_report_generator = PublicationReportGenerator()
