"""
runtime/trajectory_branching.py

Explores alternate reasoning trajectories to escape collapse basins.
Spawns parallel 'branches' of latent states and selects the most stable survivor.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

class TrajectoryBranchingEngine:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.num_branches = self.config.get("num_branches", 3)
        self.branch_temperature = self.config.get("branch_temperature", 0.5)
        self.survival_threshold = self.config.get("survival_threshold", 0.7)

    def spawn_branches(self, 
                      current_hidden: torch.Tensor, 
                      kv_states: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Dict[str, Any]]:
        """
        Creates multiple slightly perturbed versions of the current latent state.
        """
        branches = []
        
        # Branch 0 is always the original (baseline)
        branches.append({
            "id": 0,
            "hidden": current_hidden.clone(),
            "kv": kv_states,
            "score": 1.0,
            "is_original": True
        })
        
        for i in range(1, self.num_branches):
            # Apply sparse noise or manifold-aligned perturbation
            # In a real model, we might slightly change the KV attention mask or apply a different LCG guard
            noise = torch.randn_like(current_hidden) * self.branch_temperature
            branch_hidden = current_hidden + noise
            
            branches.append({
                "id": i,
                "hidden": branch_hidden,
                "kv": kv_states, # For now sharing KV, but could branch KV too
                "score": 0.0,
                "is_original": False
            })
            
        return branches

    def score_branch_survival(self, 
                              branch_results: List[Dict[str, Any]], 
                              health_fn) -> int:
        """
        Evaluates the health of each branch after one or more steps.
        Returns the index of the best branch.
        """
        best_idx = 0
        best_score = -1.0
        
        for i, branch in enumerate(branch_results):
            # The health_fn would be the CognitiveStateEngine's assessment of the new state
            score = branch.get("health_score", 0.0)
            diversity_penalty = 0.0
            
            # Bonus for surviving branches that are different from original if original is failing
            if not branch["is_original"] and branch_results[0].get("health_score", 0.0) < 0.4:
                diversity_penalty = 0.1 # Actually a bonus in this logic
                
            final_score = score + diversity_penalty
            branch["score"] = final_score
            
            if final_score > best_score:
                best_score = final_score
                best_idx = i
                
        return best_idx

    def calculate_branch_diversity(self, branches: List[Dict[str, Any]]) -> float:
        """
        Measures how different the branches are from each other.
        """
        if len(branches) < 2:
            return 0.0
            
        hiddens = torch.stack([b["hidden"] for b in branches])
        # Average pairwise distance
        dist = torch.pdist(hiddens.view(len(branches), -1)).mean().item()
        return dist

    def prune_branches(self, branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes branches that have collapsed.
        """
        return [b for b in branches if b.get("score", 1.0) > self.survival_threshold]

    def select_best_branch(self, branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Picks the best branch and returns its state.
        """
        best_branch = max(branches, key=lambda x: x.get("score", 0.0))
        return best_branch
