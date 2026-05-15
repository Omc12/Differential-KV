"""
benchmarks/obs/benchmark_workload_registry.py

Standardized benchmark workload registry for Differential KV.
Categorizes prompts by context length and type.
"""

import json
import random
from typing import List, Dict, Any, Optional

class BenchmarkWorkloadRegistry:
    """
    Registry of standardized workloads for operational benchmarking.
    """
    def __init__(self):
        self.workloads = {
            "short": [
                {"name": "General Chat", "prompt": "Explain the concept of entropy in 50 words.", "max_tokens": 50},
                {"name": "Python Snippet", "prompt": "Write a Python function to calculate Fibonacci numbers.", "max_tokens": 100}
            ],
            "medium": [
                {"name": "Technical Summary", "prompt": "Summarize the following architecture: " + ("KV cache " * 500), "max_tokens": 150},
                {"name": "Multi-Step Reasoning", "prompt": "If a train leaves at 5pm and travels at 60mph...", "max_tokens": 200}
            ],
            "long": [
                {"name": "Long Document Retrieval", "prompt": "Based on the 10,000 word document below, who is the protagonist? " + ("Protagonist " * 2000), "max_tokens": 50},
                {"name": "Complex Codebase Analysis", "prompt": "Analyze the following 50 files for security vulnerabilities: " + ("import os; " * 1000), "max_tokens": 300}
            ]
        }

    def get_workload_suite(self, category: str = "all") -> List[Dict[str, Any]]:
        """Returns a list of workloads for the given category."""
        if category == "all":
            all_workloads = []
            for cat in self.workloads.values():
                all_workloads.extend(cat)
            return all_workloads
        return self.workloads.get(category, [])

    def create_synthetic_workload(self, context_len: int, name: str = "synthetic") -> Dict[str, Any]:
        """Creates a synthetic workload of a specific length."""
        tokens = ["token"] * context_len
        prompt = " ".join(tokens)
        return {
            "name": f"{name}_{context_len}",
            "prompt": prompt,
            "max_tokens": 50
        }

if __name__ == "__main__":
    registry = BenchmarkWorkloadRegistry()
    print(f"Loaded {len(registry.get_workload_suite())} workloads.")
