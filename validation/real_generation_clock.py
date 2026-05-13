import time

class RealGenerationClock:
    """
    A monotonic clock specifically for generation tasks.
    Avoids system clock drift and provides high-resolution wall-clock timing.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_wall = time.time()
        self.start_perf = time.perf_counter()

    def elapsed_wall(self):
        return time.time() - self.start_wall

    def elapsed_perf(self):
        return time.perf_counter() - self.start_perf

    def get_timestamp(self):
        return datetime.now().isoformat()
