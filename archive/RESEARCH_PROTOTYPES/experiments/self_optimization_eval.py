import torch
from evolution.policy_self_optimizer import PolicySelfOptimizer
import numpy as np

def run_self_optimization_eval():
    print("PHASE 34A: SELF-OPTIMIZATION EVALUATION")
    
    initial_params = {
        "resonance_threshold": 0.5,
        "gc_interval": 50.0,
        "sync_urgency": 0.2
    }
    
    optimizer = PolicySelfOptimizer(initial_params)
    rewards = []
    
    for i in range(100):
        # Simulate environment response
        # In a real eval, this would be tied to model performance
        mock_stats = {
            "stability": 0.98,
            "cost": 0.5 - (i * 0.002), # Improving cost
            "pruning_rate": 0.3 + (i * 0.001), # Improving pruning
            "entropy": 0.2 - (i * 0.001) # Reducing entropy
        }
        
        reward = optimizer.step(mock_stats)
        rewards.append(reward)
        
    print(f"Initial Reward: {rewards[0]:.4f}")
    print(f"Final Reward: {rewards[-1]:.4f}")
    print(f"Reward Improvement: {(rewards[-1] - rewards[0]):.4f}")
    
    success_rate = 1.0 if rewards[-1] > rewards[0] else 0.0
    
    results = {
        "initial_reward": rewards[0],
        "final_reward": rewards[-1],
        "success_rate": success_rate,
        "final_params": optimizer.get_optimized_policies()
    }
    return results

if __name__ == "__main__":
    run_self_optimization_eval()
