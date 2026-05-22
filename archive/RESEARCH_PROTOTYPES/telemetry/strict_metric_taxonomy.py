class StrictMetricTaxonomy:
    """
    PHASE 18.1E: Enforces the mandatory taxonomy for all Phase 18.1 metrics.
    [MEASURED], [ESTIMATED], [PROJECTED], [SIMULATED]
    """
    def __init__(self):
        self.MEASURED = "[MEASURED]"
        self.ESTIMATED = "[ESTIMATED]"
        self.PROJECTED = "[PROJECTED]"
        self.SIMULATED = "[SIMULATED]"

    def format_metric(self, name: str, value: any, unit: str, taxonomy: str):
        if taxonomy not in [self.MEASURED, self.ESTIMATED, self.PROJECTED, self.SIMULATED]:
            raise ValueError(f"Invalid taxonomy label: {taxonomy}")
        
        return f"{name}: {taxonomy} {value} {unit}"

    def log_measured(self, name: str, value: any, unit: str = ""):
        return self.format_metric(name, value, unit, self.MEASURED)

    def log_estimated(self, name: str, value: any, unit: str = ""):
        return self.format_metric(name, value, unit, self.ESTIMATED)
