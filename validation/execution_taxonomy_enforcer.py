class ExecutionTaxonomyEnforcer:
    """
    Enforces the mandatory Phase 18 taxonomy for all reported metrics.
    [MEASURED], [ESTIMATED], [PROJECTED], [SIMULATED]
    """
    def __init__(self):
        self.valid_labels = ["[MEASURED]", "[ESTIMATED]", "[PROJECTED]", "[SIMULATED]"]

    def validate_report(self, report_content: str):
        """Checks if all metrics in the report have a valid taxonomy label."""
        # Simple heuristic: check if at least one label exists for each line containing 'TPS' or 'Latency'
        lines = report_content.split('\n')
        violations = []
        
        for i, line in enumerate(lines):
            if any(key in line for key in ["TPS", "Latency", "VRAM", "VRAM Residency"]):
                if not any(label in line for label in self.valid_labels):
                    violations.append(f"Line {i+1}: Missing taxonomy label for metric. Content: '{line.strip()}'")
        
        return len(violations) == 0, violations

    def label_measured(self, value):
        return f"[MEASURED] {value}"

    def label_estimated(self, value):
        return f"[ESTIMATED] {value}"
