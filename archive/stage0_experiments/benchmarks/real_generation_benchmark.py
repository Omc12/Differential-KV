import time
import torch
from typing import List, Dict, Any

class RealGenerationBenchmark:
    """
    Benchmarks real transformer generation.
    Measures actual tokens/sec and latency based on genuine model execution.
    """
    def __init__(self, engine):
        self.engine = engine

    def run_benchmark(self, prompts: List[str], max_new_tokens: int = 50) -> List[Dict[str, Any]]:
        results = []
        for prompt in prompts:
            start_time = time.perf_counter()
            output = self.engine.generate(prompt, max_new_tokens=max_new_tokens)
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            if isinstance(output, dict):
                token_count = len(output.get("tokens", []))
                text = output.get("text", "")
            else:
                # Assume output is a string
                text = output
                token_count = len(text.split()) # Approximation if tokens are not returned
            
            tps = token_count / latency if latency > 0 else 0
            
            results.append({
                "prompt_len": len(prompt),
                "output_len": token_count,
                "latency": latency,
                "tps": tps,
                "text": text
            })
        return results
