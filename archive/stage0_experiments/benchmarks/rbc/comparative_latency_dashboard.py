"""
benchmarks/rbc/comparative_latency_dashboard.py

Generates comparative latency dashboards and metrics.
Compares TTFT, ITL, and TPS across runtimes.
"""

from typing import List, Dict, Any

class ComparativeLatencyDashboard:
    """
    Analyzes and formats comparative latency data.
    """
    def __init__(self):
        self.data = {}

    def add_result(self, runtime: str, scenario: str, metrics: Dict[str, Any]):
        """Adds a result to the dashboard dataset."""
        if runtime not in self.data:
            self.data[runtime] = {}
        self.data[runtime][scenario] = metrics

    def generate_summary_table(self) -> str:
        """Generates a markdown table of comparative performance."""
        runtimes = list(self.data.keys())
        scenarios = set()
        for r in runtimes:
            scenarios.update(self.data[r].keys())
        
        headers = ["Scenario", "Metric"] + runtimes
        rows = []
        
        for scenario in sorted(scenarios):
            # TPS Row
            tps_row = [scenario, "TPS"]
            for r in runtimes:
                tps_row.append(f"{self.data[r].get(scenario, {}).get('tps', 0):.2f}")
            rows.append(tps_row)
            
            # TTFT Row
            ttft_row = ["", "TTFT (ms)"]
            for r in runtimes:
                ttft_row.append(f"{self.data[r].get(scenario, {}).get('ttft_ms', 0):.1f}")
            rows.append(ttft_row)
            
        # Format as Markdown
        table = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---']*len(headers))} |"]
        for row in rows:
            table.append(f"| {' | '.join(row)} |")
            
        return "\n".join(table)

if __name__ == "__main__":
    dashboard = ComparativeLatencyDashboard()
    dashboard.add_result("diff_kv", "short", {"tps": 85.0, "ttft_ms": 50.0})
    dashboard.add_result("transformers", "short", {"tps": 10.0, "ttft_ms": 200.0})
    print(dashboard.generate_summary_table())
