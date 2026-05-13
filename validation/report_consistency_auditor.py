class ReportConsistencyAuditor:
    """
    Audits multiple reports for internal consistency.
    Detects if the same model has wildly different baselines across different files.
    """
    def __init__(self):
        self.history = {}

    def audit_metric(self, model_id, metric_name, value):
        key = f"{model_id}:{metric_name}"
        if key in self.history:
            prev_value = self.history[key]
            drift = abs(value - prev_value) / prev_value if prev_value > 0 else 0
            if drift > 0.2:
                return False, f"Inconsistency detected for {key}: drift={drift:.2%}"
        
        self.history[key] = value
        return True, "OK"
