"""
analysis/evolutionary_attractor_optimizer.py
Phase 18: Evolutionary Manifold Shaping
Scores attractors, merges redundant basins, and tracks evolution across generations.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from analysis.basin_fitness import BasinFitnessEvaluator

class EvolutionaryAttractorOptimizer:
    def __init__(self, fitness_evaluator: BasinFitnessEvaluator):
        self.fitness_evaluator = fitness_evaluator
        self.generation_history = []
        self.active_basins = [] # List of basin metadata

    def score_population(self, population_trajectories: List[Dict]) -> List[Dict]:
        """
        Scores a generation of manifold trajectories.
        """
        scored_population = []
        for traj_data in population_trajectories:
            fitness = self.fitness_evaluator.compute_fitness(
                trajectory=traj_data["hidden_states"],
                logits=traj_data["logits"],
                retrieval_accuracies=traj_data.get("retrieval_accs", []),
                recovery_events=traj_data.get("recovery_events", [])
            )
            traj_data["fitness"] = fitness
            scored_population.append(traj_data)
            
        self.generation_history.append(scored_population)
        return scored_population

    def merge_redundant_basins(self, population: List[Dict], threshold: float = 0.95) -> List[Dict]:
        """
        Identifies and merges basins that are geometrically similar (high cosine similarity).
        """
        if not population: return []
        
        merged = []
        visited = set()
        
        for i in range(len(population)):
            if i in visited: continue
            
            current_basin = population[i]
            # Represent basin by its mean state
            mean_state_i = torch.stack(current_basin["hidden_states"]).mean(dim=0)
            
            to_merge = [i]
            for j in range(i + 1, len(population)):
                if j in visited: continue
                
                mean_state_j = torch.stack(population[j]["hidden_states"]).mean(dim=0)
                sim = torch.nn.functional.cosine_similarity(mean_state_i.unsqueeze(0), mean_state_j.unsqueeze(0)).item()
                
                if sim > threshold:
                    to_merge.append(j)
                    visited.add(j)
            
            # Simple merge: keep the one with highest fitness
            best_idx = max(to_merge, key=lambda idx: population[idx]["fitness"]["unified_fitness"])
            merged.append(population[best_idx])
            visited.add(i)
            
        return merged

    def measure_manifold_evolution(self) -> Dict[str, List[float]]:
        """
        Tracks metrics like collapse resistance and manifold smoothness over generations.
        """
        metrics = {
            "avg_fitness": [],
            "collapse_resistance": [], # Percentage of non-collapsed basins
            "manifold_smoothness": [],
            "basin_diversity": []
        }
        
        for gen in self.generation_history:
            fitnesses = [p["fitness"]["unified_fitness"] for p in gen]
            metrics["avg_fitness"].append(np.mean(fitnesses))
            
            non_collapsed = [p for p in gen if p["fitness"]["recovery_prob"] > 0.5]
            metrics["collapse_resistance"].append(len(non_collapsed) / len(gen))
            
            smoothness = [p["fitness"]["smoothness"] for p in gen]
            metrics["manifold_smoothness"].append(np.mean(smoothness))
            
            # Diversity: average pairwise distance between mean states
            means = [torch.stack(p["hidden_states"]).mean(dim=0) for p in gen]
            if len(means) > 1:
                dist = 0
                count = 0
                for i in range(len(means)):
                    for j in range(i+1, len(means)):
                        dist += torch.norm(means[i] - means[j], p=2).item()
                        count += 1
                metrics["basin_diversity"].append(dist / count)
            else:
                metrics["basin_diversity"].append(0.0)
                
        return metrics
