"""
benchmarks/rbc/standardized_benchmark_matrix.py

Defines standardized configurations for comparative benchmarking.
Ensures apples-to-apples comparisons across runtimes.
"""

from typing import List, Dict, Any

class StandardizedBenchmarkMatrix:
    """
    Registry of standardized benchmark configurations.
    """
    def __init__(self):
        self.matrix = {
            "scenarios": [
                {"name": "interactive_short", "context_len": 512, "gen_len": 50, "concurrency": 1},
                {"name": "document_qa_medium", "context_len": 4096, "gen_len": 200, "concurrency": 1},
                {"name": "code_analysis_long", "context_len": 16384, "gen_len": 500, "concurrency": 1},
                {"name": "high_concurrency_serving", "context_len": 1024, "gen_len": 100, "concurrency": 16}
            ]
        }

    def get_scenarios(self) -> List[Dict[str, Any]]:
        """Returns the list of standardized scenarios."""
        return self.matrix["scenarios"]

    def get_config_for_scenario(self, name: str) -> Dict[str, Any]:
        """Returns the configuration for a specific scenario."""
        for scenario in self.matrix["scenarios"]:
            if scenario["name"] == name:
                return scenario
        return {}

if __name__ == "__main__":
    matrix = StandardizedBenchmarkMatrix()
    print(f"Matrix loaded with {len(matrix.get_scenarios())} scenarios.")
