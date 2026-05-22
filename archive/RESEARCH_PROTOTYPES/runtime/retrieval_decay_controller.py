class RetrievalDecayController:
    """
    Adaptive logic to recover density if retrieval quality collapses.
    """
    def __init__(self, target_score: float = 0.9):
        self.target_score = target_score

    def should_adjust(self, current_score: float) -> bool:
        return current_score < self.target_score

    def get_adjustment(self) -> str:
        """
        Returns an adjustment command (e.g., increase density).
        """
        return "INCREASE_DENSITY"
