import time

class SalienceOverheadTracker:
    """
    PHASE 20.1E: Tracks the compute overhead of salience modeling and importance propagation.
    Ensures that ASSCIM remains practical for real-time sparse inference.
    """
    def __init__(self):
        self.total_time = 0.0
        self.step_count = 0
        self.history = []

    def record(self, duration: float):
        self.total_time += duration
        self.step_count += 1
        self.history.append(duration)

    def get_avg_overhead_ms(self) -> float:
        if self.step_count == 0:
            return 0.0
        return (self.total_time / self.step_count) * 1000.0

    def get_summary(self):
        return {
            "avg_overhead_ms": self.get_avg_overhead_ms(),
            "max_overhead_ms": max(self.history) * 1000.0 if self.history else 0.0,
            "total_steps": self.step_count
        }
