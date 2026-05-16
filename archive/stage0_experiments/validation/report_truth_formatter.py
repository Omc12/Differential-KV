class ReportTruthFormatter:
    """
    Formats validation reports with a focus on scientific truth and defensibility.
    Adds mandatory metadata to every reported metric.
    """
    def __init__(self):
        pass

    def format_metric(self, name, value, confidence="High", unit="TPS"):
        return {
            "name": name,
            "value": value,
            "unit": unit,
            "confidence": confidence,
            "physically_plausible": "YES" if value < 10000 else "NO" # Mock check
        }

    def generate_summary_table(self, metrics: list):
        header = "| Metric | Value | Unit | Confidence | Plausible |\n|---|---|---|---|---|"
        rows = []
        for m in metrics:
            rows.append(f"| {m['name']} | {m['value']:.2f} | {m['unit']} | {m['confidence']} | {m['physically_plausible']} |")
        return header + "\n" + "\n".join(rows)
