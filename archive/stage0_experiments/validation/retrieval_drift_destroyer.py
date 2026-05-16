class RetrievalDriftDestroyer:
    """
    Rejects runs with excessive retrieval drift or collapse.
    """
    def __init__(self, max_drift: float = 0.05):
        self.max_drift = max_drift

    def audit_drift(self, drift: float):
        if abs(drift) > self.max_drift:
            raise ValueError(f"CRITICAL FAILURE: Retrieval drift {drift:.4f} exceeds threshold {self.max_drift}")
        return True
