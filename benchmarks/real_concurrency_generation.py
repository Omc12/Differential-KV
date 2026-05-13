import threading
import time
from typing import List

class RealConcurrencyGeneration:
    """
    Simulates real concurrent multi-user generation.
    Tests how the sparse runtime handles multiple active KV streams.
    """
    def __init__(self, engine):
        self.engine = engine

    def run_concurrent_test(self, prompts: List[str], max_new_tokens: int = 50):
        results = []
        threads = []

        def worker(prompt):
            start = time.time()
            out = self.engine.generate(prompt, max_new_tokens=max_new_tokens)
            latency = time.time() - start
            results.append({
                "prompt": prompt[:30],
                "latency": latency,
                "tokens": len(out["tokens"])
            })

        for p in prompts:
            t = threading.Thread(target=worker, args=(p,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results
