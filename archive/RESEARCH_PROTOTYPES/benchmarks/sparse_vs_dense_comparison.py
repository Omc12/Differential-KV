import time
from typing import List, Dict, Any

class SparseVsDenseComparison:
    """
    Direct comparison between standard dense inference and Differential KV sparse inference.
    Compares speed, memory, and output quality.
    """
    def __init__(self, dense_engine, sparse_engine):
        self.dense_engine = dense_engine
        self.sparse_engine = sparse_engine

    def run_comparison(self, prompt: str, max_new_tokens: int = 100):
        # Run Dense
        start = time.perf_counter()
        dense_out = self.dense_engine.generate(prompt, max_new_tokens=max_new_tokens)
        dense_time = time.perf_counter() - start
        
        # Run Sparse
        start = time.perf_counter()
        sparse_out = self.sparse_engine.generate(prompt, max_new_tokens=max_new_tokens)
        sparse_time = time.perf_counter() - start
        
        return {
            "dense": {
                "latency": dense_time,
                "tps": len(dense_out["tokens"]) / dense_time,
                "text": dense_out["text"]
            },
            "sparse": {
                "latency": sparse_time,
                "tps": len(sparse_out["tokens"]) / sparse_time,
                "text": sparse_out["text"]
            },
            "speedup": dense_time / sparse_time if sparse_time > 0 else 0
        }
