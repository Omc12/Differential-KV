import time

class DecoderOverheadTracker:
    def __init__(self):
        self.total_arbitration_time = 0.0
    def record(self, duration):
        self.total_arbitration_time += duration
