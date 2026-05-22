import time
import torch

class WallclockEnforcer:
    """
    PHASE 18.1E: Ensures all timing is actual hardware wall-clock time.
    FORBIDDEN: Simulated or interpolated timing.
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time

    def get_timestamp(self):
        return time.time()
