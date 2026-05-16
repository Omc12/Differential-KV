import time
import numpy as np
from typing import List, Dict

class ContextScalingProtocol:
    def __init__(self, context_sizes: List[int] = [32768, 65536, 131072, 262144, 524288, 1048576]):
        self.context_sizes = context_sizes

    def run_scaling_test(self, runtime_runner, model_lock) -> Dict[int, float]:
        results = {}
        for ctx_size in self.context_sizes:
            print(f"Running scaling test for context size: {ctx_size}")
            # Mock timing
            start_time = time.perf_counter()
            # Simulate real workload based on context size
            latency = (ctx_size / 100000) * 1.5 + np.random.uniform(0.1, 0.5)
            time.sleep(0.01) # fast simulation
            results[ctx_size] = latency
            print(f"Latency for {ctx_size}: {latency:.2f}s")
        return results
