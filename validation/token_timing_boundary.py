import time

class TokenTimingBoundary:
    """
    Defines strict boundaries for token timing measurement.
    Ensures that only the actual model forward pass is counted in generation TPS.
    """
    def __init__(self):
        self.boundaries = {}

    def start_boundary(self, name):
        self.boundaries[name] = {"start": time.perf_counter()}

    def end_boundary(self, name):
        if name in self.boundaries:
            self.boundaries[name]["end"] = time.perf_counter()
            self.boundaries[name]["duration"] = self.boundaries[name]["end"] - self.boundaries[name]["start"]
            return self.boundaries[name]["duration"]
        return 0

    def get_breakdown(self):
        return {k: v.get("duration", 0) for k, v in self.boundaries.items()}
