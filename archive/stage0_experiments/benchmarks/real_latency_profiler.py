import time

class RealLatencyProfiler:
    """
    Profiles the latency of each component in the real inference path.
    Identifies where the time is actually spent.
    """
    def __init__(self):
        self.points = {}

    def start(self, label):
        self.points[label] = time.perf_counter()

    def stop(self, label):
        if label in self.points:
            duration = time.perf_counter() - self.points[label]
            return duration
        return 0

    def profile_call(self, func, *args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        duration = time.perf_counter() - start
        return res, duration
