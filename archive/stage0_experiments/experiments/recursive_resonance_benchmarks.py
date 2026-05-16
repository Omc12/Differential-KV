import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from experiments.infinite_horizon_reasoning import InfiniteHorizonReasoningSim
from experiments.persistent_cognition_eval import PersistentCognitionEval

class RecursiveResonanceBenchmarks:
    """
    Main benchmark suite for PHASE 25 - Recursive Cognitive Resonance.
    """
    def __init__(self):
        self.results = []

    def run_all(self):
        print("=== PHASE 25 RECURSIVE RESONANCE BENCHMARKS ===")
        
        # 1. Infinite Horizon Reasoning (100k+ reasoning steps simulated)
        # We scale the simulation to 1000 steps but interpret it as long-range CoT
        sim = InfiniteHorizonReasoningSim(max_steps=1000)
        
        print("\nRunning Baseline...")
        baseline = sim.run_experiment(use_rcr=False)
        
        print("\nRunning RCR...")
        rcr = sim.run_experiment(use_rcr=True)
        
        # 2. Persistent Cognition
        peval = PersistentCognitionEval()
        presults = peval.run_eval()
        
        # 3. Compile Comparison
        summary = {
            "Metric": [
                "Reasoning Survival (steps)",
                "Survival Ratio",
                "Throughput (steps/sec)",
                "Context Fidelity",
                "Memory Overhead (MB)"
            ],
            "Baseline": [
                baseline['survival_steps'],
                baseline['survival_ratio'],
                baseline['throughput'],
                "N/A",
                0.0
            ],
            "RCR": [
                rcr['survival_steps'],
                rcr['survival_ratio'],
                rcr['throughput'],
                presults['state_fidelity'],
                presults['overhead_mb']
            ]
        }
        
        df = pd.DataFrame(summary)
        print("\nSummary Results:")
        print(df)
        
        # Performance Impact
        throughput_penalty = (1 - (rcr['throughput'] / baseline['throughput'])) * 100
        print(f"\nThroughput Penalty: {throughput_penalty:.2f}%")
        
        # Survival Gain
        survival_gain = (rcr['survival_steps'] / (baseline['survival_steps'] + 1e-6))
        print(f"Reasoning Survival Gain: {survival_gain:.2f}x")
        
        return df

if __name__ == "__main__":
    bench = RecursiveResonanceBenchmarks()
    bench.run_all()
