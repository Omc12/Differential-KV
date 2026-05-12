import torch
from typing import Dict

class RuntimeRewardModel:
    """
    Evaluates runtime efficiency and stabilization success to provide a reward signal.
    Used for reinforcement-based policy tuning.
    """
    def __init__(self):
        self.last_stats = None
        
    def calculate_reward(self, current_stats: Dict) -> float:
        """
        Calculates a reward based on:
        - Stabilization cost reduction (positive)
        - Memory efficiency (positive)
        - Entropy reduction (positive)
        - Throughput (positive)
        - Accuracy/Stability (critical constraint)
        """
        # stats keys: cost, memory, entropy, throughput, stability
        
        # Penalize instability heavily
        if current_stats["stability"] < 0.95:
            return -10.0
            
        # Reward cost reduction
        cost_reward = 1.0 / (current_stats["cost"] + 1e-6)
        
        # Reward memory pruning
        memory_reward = current_stats["pruning_rate"] * 2.0
        
        # Reward low entropy
        entropy_reward = max(0, 1.0 - current_stats["entropy"])
        
        total_reward = (cost_reward * 0.4) + (memory_reward * 0.3) + (entropy_reward * 0.3)
        return total_reward

    def get_performance_delta(self, current_stats: Dict) -> float:
        if self.last_stats is None:
            self.last_stats = current_stats
            return 0.0
            
        delta = current_stats["stability"] - self.last_stats["stability"]
        self.last_stats = current_stats
        return delta
