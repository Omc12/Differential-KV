"""
benchmarks/universality_benchmarks.py
Phase 19: Universal Cognitive Geometry
Comprehensive benchmark suite for cross-model universality.
"""

import os
import json
import torch
from typing import List, Dict, Any
from analysis.universal_collapse_signatures import CollapseSignatureAnalyzer
from analysis.geometry_invariant_reasoning import ReasoningGeometryInvariance
from analysis.cognitive_phase_diagrams import CognitivePhaseDiagram
from experiments.self_compressing_cognition import SelfCompressingCognitionExperiment
from experiments.cross_model_anchor_transfer import AnchorTransferExperiment

class UniversalityBenchmarkSuite:
    def __init__(self, models: List[str] = ["Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]):
        self.models = models
        self.results = {}

    def run_all(self):
        print("Starting Phase 19 Universality Benchmarks...")
        
        # 1. Reasoning Invariance
        print("Benchmarking Reasoning Invariance...")
        rgi = ReasoningGeometryInvariance()
        # Mock tasks for brevity in benchmark script
        tasks = {
            "arithmetic": "Calculate 123 * 456 step by step.",
            "coding": "Write a python function to find primes.",
            "planning": "Plan a trip from London to Tokyo."
        }
        
        # For the benchmark, we'll use the first model
        from analysis.reasoning_manifold import ReasoningTrajectoryTracker
        tracker = ReasoningTrajectoryTracker(model_id=self.models[0])
        
        for name, prompt in tasks.items():
            _, traj = tracker.run_generation(prompt, max_new_tokens=20)
            states = [t["hidden"][-1][0, -1, :].numpy() for t in traj]
            rgi.record_task_geometry(name, states)
            
        invariants = rgi.compare_invariants()
        self.results["reasoning_invariance"] = invariants
        
        # 2. Universal Collapse Signatures
        print("Benchmarking Universal Collapse...")
        analyzer = CollapseSignatureAnalyzer()
        # Simulate a collapsing run with high noise
        def high_noise_mod(l, k, v):
            return k + torch.randn_like(k) * 0.5, v + torch.randn_like(v) * 0.5
            
        _, traj_collapse = tracker.run_generation(tasks["arithmetic"], kv_modifier_fn=high_noise_mod, max_new_tokens=20)
        profile = analyzer.build_collapse_profile({"traj": traj_collapse})
        self.results["collapse_signatures"] = profile
        
        # 3. Anchor Transfer
        print("Benchmarking Anchor Transfer...")
        if len(self.models) >= 2:
            ate = AnchorTransferExperiment(self.models[0], self.models[1])
            transfer_res = ate.run_transfer("What is the capital of France?")
            self.results["anchor_transfer"] = transfer_res
            
        # 4. Self-Compression
        print("Benchmarking Self-Compression...")
        sce = SelfCompressingCognitionExperiment(model_id=self.models[0])
        comp_res = sce.run_self_compression_test("Explain gravity.")
        self.results["self_compression"] = comp_res
        
        self.save_results()

    def save_results(self):
        os.makedirs("results/phase19", exist_ok=True)
        with open("results/phase19/universality_benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=4)
        print("Results saved to results/phase19/universality_benchmark_results.json")

if __name__ == "__main__":
    suite = UniversalityBenchmarkSuite()
    suite.run_all()
