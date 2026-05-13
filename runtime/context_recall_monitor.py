import random

class ContextRecallMonitor:
    """
    Performs periodic "needle" tests during long runs to verify recall.
    """
    def __init__(self, interval_steps: int = 100):
        self.interval = interval_steps
        self.results = []

    def should_test(self, step: int) -> bool:
        return step > 0 and step % self.interval == 0

    def run_recall_test(self) -> bool:
        """
        Simulates a recall test.
        In real impl, this would inject a needle and verify retrieval.
        """
        success = random.random() > 0.05 # 95% success baseline
        self.results.append(success)
        return success

    def get_recall_accuracy(self) -> float:
        if not self.results:
            return 1.0
        return sum(self.results) / len(self.results)
