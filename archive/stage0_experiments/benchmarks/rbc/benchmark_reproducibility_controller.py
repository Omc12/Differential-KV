"""
benchmarks/rbc/benchmark_reproducibility_controller.py

Controls reproducibility and scientific validity of benchmarks.
Ensures fixed seeds and repeated trials.
"""

import torch
import random
import numpy as np
from typing import List, Dict, Any

class BenchmarkReproducibilityController:
    """
    Enforces scientific controls over the benchmarking process.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def enforce_seeds(self):
        """Sets global seeds for all relevant libraries."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run_repeated_trials(self, benchmark_fn, n: int = 3) -> Dict[str, Any]:
        """
        Runs a benchmark function N times and calculates statistics.
        """
        results = []
        for i in range(n):
            self.enforce_seeds()
            results.append(benchmark_fn())
            
        tps_values = [r.get("tps", 0) for r in results]
        return {
            "mean_tps": np.mean(tps_values),
            "std_tps": np.std(tps_values),
            "variance_index": np.std(tps_values) / np.mean(tps_values) if np.mean(tps_values) > 0 else 0,
            "trials": n
        }

if __name__ == "__main__":
    controller = BenchmarkReproducibilityController()
    print(controller.run_repeated_trials(lambda: {"tps": 85.0 + random.random()}))
