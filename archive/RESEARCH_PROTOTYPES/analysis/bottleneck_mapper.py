import json
import os

class BottleneckMapper:
    """
    PHASE 18.1F: Maps failure boundaries and performance bottlenecks for real-model execution.
    """
    def __init__(self, export_path: str = "results/reconstruction_18_1/reconstruction_18_1_failure_analysis.md"):
        self.export_path = export_path
        self.bottlenecks = []
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

    def record_bottleneck(self, area: str, observed_limit: str, failure_mode: str, taxonomy: str = "[MEASURED]"):
        self.bottlenecks.append({
            "area": area,
            "limit": observed_limit,
            "mode": failure_mode,
            "taxonomy": taxonomy
        })

    def export_report(self):
        with open(self.export_path, 'w') as f:
            f.write("# PHASE 18.1 FAILURE ANALYSIS & BOTTLENECK MAPPING\n\n")
            f.write("## Observed Execution Boundaries\n\n")
            f.write("| Component | Failure/Limit | Taxonomy | Root Cause |\n")
            f.write("|---|---|---|---|\n")
            for b in self.bottlenecks:
                f.write(f"| {b['area']} | {b['limit']} | {b['taxonomy']} | {b['mode']} |\n")
            
            f.write("\n\n> [!CAUTION]\n")
            f.write("> Negative results in this phase are considered high-value scientific data for Differential KV evolution.\n")
        
        return self.export_path
