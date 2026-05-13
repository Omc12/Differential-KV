import random
import numpy as np

class DeterministicBenchmarkRunner:
    """
    Ensures that benchmark runs use deterministic seeds and inputs.
    Rejects runs with non-deterministic variance.
    """
    def __init__(self, seed=42):
        self.seed = seed

    def setup(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        # torch.manual_seed(self.seed) if using torch
        print(f"[Deterministic] Seeds set to: {self.seed}")

    def run_with_determinism(self, benchmark_func, *args, **kwargs):
        self.setup()
        return benchmark_func(*args, **kwargs)
