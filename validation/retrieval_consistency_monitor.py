class RetrievalConsistencyMonitor:
    """Monitors retrieval stability and token-level exact match consistency."""
    def __init__(self):
        self.history = []

    def log_retrieval(self, expected, generated):
        match = expected.lower().strip() in generated.lower().strip()
        self.history.append({"expected": expected, "match": match})
        return match

    def get_consistency_score(self):
        if not self.history:
            return 1.0
        matches = [h["match"] for h in self.history]
        return sum(matches) / len(matches)
