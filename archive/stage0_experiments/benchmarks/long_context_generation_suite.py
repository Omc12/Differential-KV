import time
from validation.real_inference_harness import RealInferenceHarness

class LongContextBenchmark:
    """
    Benchmarks Differential KV under long-context generation workloads.
    Focuses on context lengths from 32k to 512k+.
    """
    def __init__(self, harness):
        self.harness = harness

    def run_benchmark(self, context_lengths=[32768, 65536, 131072, 262144]):
        results = {}
        for length in context_lengths:
            print(f"[Benchmark] Testing context length: {length}")
            # Generate a prompt of specific length
            dummy_prompt = "A" * length 
            res = self.harness.execute_request(dummy_prompt, max_tokens=100)
            results[length] = res
            
        return results
