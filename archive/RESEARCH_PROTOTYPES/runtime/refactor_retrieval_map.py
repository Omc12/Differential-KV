class RefactorRetrievalMap:
    """
    Maps retrieval success rates across different phases of a refactor.
    Helps identify where in the edit cycle retrieval starts to degrade.
    """
    def __init__(self):
        self.phase_scores = {} # phase_id -> avg_score

    def log_phase(self, phase_id: str, scores: list):
        self.phase_scores[phase_id] = sum(scores) / len(scores) if scores else 0.0

    def get_degradation_point(self, threshold: float = 0.8):
        for pid, score in self.phase_scores.items():
            if score < threshold:
                return pid
        return None
