"""
experiments/cognitive_evolution_loop.py
Phase 18: Evolutionary Manifold Shaping
Runs iterative generations of cognitive evolution: mutate, evaluate, select.
"""

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
from analysis.basin_fitness import BasinFitnessEvaluator
from analysis.evolutionary_attractor_optimizer import EvolutionaryAttractorOptimizer
from anchor_logic.cognitive_mutator import CognitiveMutator
from anchor_logic.basin_reinforcement import BasinReinforcementSystem

class CognitiveEvolutionLoop:
    def __init__(self, 
                 model_id="Qwen/Qwen2-0.5B", 
                 pop_size=5, 
                 generations=3, 
                 device="cuda"):
        self.tracker = ReasoningTrajectoryTracker(model_id=model_id, device=device)
        self.fitness_evaluator = BasinFitnessEvaluator(device=device)
        self.optimizer = EvolutionaryAttractorOptimizer(self.fitness_evaluator)
        self.mutator = CognitiveMutator()
        self.pop_size = pop_size
        self.generations = generations
        
        # Initial population of configs
        self.population = [
            {"anchor_density": 0.1, "base_rank": 8, "repair_strength": 1.0, "drift_threshold": 0.05}
            for _ in range(pop_size)
        ]

    def run_generation(self, gen_idx: int, prompt: str):
        print(f"\n--- Generation {gen_idx} ---")
        population_results = []
        
        for i, config in enumerate(self.population):
            print(f"Testing Individual {i} with config: {config}")
            
            def evolutionary_mod(l_idx, k, v):
                # Apply mutations/config to the KV process
                # 1. Sparsify based on anchor_density
                seq_len = k.shape[2]
                mask = (torch.rand(seq_len) < config["anchor_density"]).to(dtype=k.dtype, device=k.device)
                # Apply noise to non-anchored tokens to simulate compression
                noise = torch.randn_like(k) * (1.0 - mask).view(1, 1, -1, 1) * 0.1
                return k + noise, v + noise

            ids, traj = self.tracker.run_generation(prompt, max_new_tokens=30, kv_modifier_fn=evolutionary_mod)
            
            # Pack results for scoring
            res = {
                "config": config,
                "hidden_states": [t["hidden"][-1][0, -1, :] for t in traj],
                "logits": [torch.randn(1, 32000) for _ in traj], # Mock logits for speed
                "retrieval_accs": [0.9] * len(traj), # Mock
                "recovery_events": [True] * len(traj) # Mock
            }
            population_results.append(res)
            
        # Score and Optimize
        scored_pop = self.optimizer.score_population(population_results)
        
        # Selection: Top 50%
        scored_pop.sort(key=lambda x: x["fitness"]["unified_fitness"], reverse=True)
        elites = scored_pop[:self.pop_size // 2]
        
        # Reproduction
        new_pop = [e["config"] for e in elites]
        while len(new_pop) < self.pop_size:
            parent_a = np.random.choice(elites)["config"]
            parent_b = np.random.choice(elites)["config"]
            child = self.mutator.crossover(parent_a, parent_b)
            child = self.mutator.mutate_config(child)
            new_pop.append(child)
            
        self.population = new_pop
        return scored_pop

    def run_evolution(self, prompt: str):
        for g in range(self.generations):
            self.run_generation(g, prompt)
            
        evolution_metrics = self.optimizer.measure_manifold_evolution()
        return evolution_metrics

    def plot_evolution(self, metrics: Dict, save_path: str):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["avg_fitness"], label="Avg Unified Fitness", marker='o')
        plt.plot(metrics["manifold_smoothness"], label="Manifold Smoothness", marker='s')
        plt.plot(metrics["collapse_resistance"], label="Collapse Resistance", marker='^')
        plt.title("Evolution of Cognitive Manifolds")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    loop = CognitiveEvolutionLoop(generations=5, pop_size=6)
    prompt = "Step by step, prove that there are infinitely many prime numbers."
    metrics = loop.run_evolution(prompt)
    
    os.makedirs("results/phase18/plots", exist_ok=True)
    loop.plot_evolution(metrics, "results/phase18/plots/manifold_evolution.png")
    
    with open("results/phase18/evolution_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
