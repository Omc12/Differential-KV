import time

class BenchmarkBoundaryValidator:
    """
    Validates that benchmark timing boundaries are correctly placed.
    Ensures that start/end calls bracket the intended workload.
    """
    def __init__(self):
        self.timers = {}

    def start(self, label):
        self.timers[label] = time.perf_counter()

    def stop(self, label, expected_min_duration=0.0):
        if label not in self.timers:
            raise ValueError(f"Timer {label} was never started")
        
        duration = time.perf_counter() - self.timers[label]
        if duration < expected_min_duration:
            return False, f"Timer {label} duration {duration:.6f}s below minimum {expected_min_duration}s"
        return True, duration
