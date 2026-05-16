"""
canonical_benchmark_registry.py

Standardized benchmark registry for Phase 33.0 CBP.
Defines fixed workloads, concurrency matrices, and context matrices.
"""

from typing import List, Dict, Any

class CanonicalBenchmarkRegistry:
    """
    Registry of standardized benchmark definitions for publication-quality results.
    """
    
    MODELS = ["Qwen2.5-0.5B-Instruct"]
    CONTEXTS = [512, 4096]
    CONCURRENCIES = [1, 4, 8]
    GENERATION_LENGTHS = [128]
    
    def __init__(self):
        # Base workloads based on context requirements
        self.base_workloads = {
            512: "Explain the architectural differences between dense and sparse attention.",
            4096: "Summarize the following technical document regarding KV cache optimization: " + ("KV optimization " * 400),
            16384: "Analyze the long-range dependencies in this extended sequence: " + ("sequence data " * 1600),
            32768: "Perform needle-in-a-haystack retrieval on this massive context: " + ("irrelevant data " * 3200) + " The needle is: BLUE DIAMOND. " + ("more data " * 100)
        }

    def get_full_matrix(self) -> List[Dict[str, Any]]:
        """
        Returns the complete canonical benchmark matrix.
        """
        matrix = []
        for model in self.MODELS:
            for context in self.CONTEXTS:
                for concurrency in self.CONCURRENCIES:
                    for gen_len in self.GENERATION_LENGTHS:
                        matrix.append({
                            "name": f"{model}_{context}_{concurrency}_{gen_len}",
                            "model": model,
                            "context_len": context,
                            "concurrency": concurrency,
                            "gen_len": gen_len,
                            "prompt": self.base_workloads.get(context, "Default prompt")
                        })
        return matrix

    def get_reporting_schema(self) -> Dict[str, Any]:
        """
        Returns the fixed reporting schema for CBP.
        """
        return {
            "metrics": [
                "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                "ttft_ms", "itl_ms", "sustained_tps", "user_tps", "total_system_tps",
                "vram_residency_mb", "kv_residency_mb", "occupancy_stability",
                "launch_overhead_ratio", "serving_overhead_ratio",
                "sparse_runtime_pct", "dense_runtime_pct"
            ],
            "metadata": [
                "timestamp", "model_id", "hardware_id", "reproducibility_seed", "trial_count"
            ]
        }

# Global instance
benchmark_registry = CanonicalBenchmarkRegistry()
