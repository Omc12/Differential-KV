import torch

class KernelEfficiencyDashboard:
    """
    Visualizes kernel efficiency metrics for the reconstruction report.
    Compares fused sparse vs. standard dense performance.
    """
    def __init__(self):
        self.metrics = {}

    def update_metrics(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def generate_summary_table(self):
        summary = "| Metric | Mean Value |\n|---|---|\n"
        for name, values in self.metrics.items():
            mean_val = sum(values) / len(values)
            summary += f"| {name} | {mean_val:.4f} |\n"
        return summary
