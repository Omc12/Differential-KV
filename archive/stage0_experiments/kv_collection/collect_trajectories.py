"""
kv_collection/collect_trajectories.py
Phase 16: Collapse Trajectory Dataset Collection
Generates a dataset of successful and failed reasoning trajectories.
"""

import os
import torch
import json
import numpy as np
from tqdm import tqdm
from experiments.phase15_actr_validation import ACTRExperiment

def collect_data(num_samples=10):
    prompts = [
        "Question: If a train travels at 60 mph for 2 hours and then at 80 mph for 3 hours, what is the total distance traveled? Let's think step by step.",
        "Solve the following logic puzzle: A is taller than B, B is shorter than C, and C is taller than A. Who is the tallest?",
        "Explain the concept of quantum entanglement to a five-year-old.",
        "Write a python function to find the nth Fibonacci number using recursion.",
        "Summarize the main events of the French Revolution in three bullet points."
    ]
    
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2] # From perfect to collapse-inducing
    dataset = []
    
    os.makedirs("results/phase16/trajectories", exist_ok=True)
    
    for p_idx, prompt in enumerate(prompts):
        for noise in noise_levels:
            print(f"Collecting: Prompt {p_idx}, Noise {noise}")
            exp = ACTRExperiment()
            
            # Run without ACTR to observe collapse patterns
            res = exp.run_experiment(prompt, noise_std=noise, use_actr=False)
            
            trajectory = {
                "prompt_id": p_idx,
                "noise_level": noise,
                "text": res["text"],
                "metrics": res["metrics"],
                "is_collapsed": noise > 0.1 # Heuristic for labeling
            }
            dataset.append(trajectory)
            
            # Save individual trajectory
            filename = f"results/phase16/trajectories/traj_p{p_idx}_n{noise:.2f}.json"
            with open(filename, "w") as f:
                json.dump(trajectory, f, indent=4)
                
    return dataset

if __name__ == "__main__":
    # We'll run a small collection for demonstration
    # Note: This requires a GPU and the Qwen2 model
    try:
        data = collect_data(num_samples=2)
        print(f"Data collection complete. {len(data)} trajectories saved.")
    except Exception as e:
        print(f"Data collection failed (likely missing GPU/model): {e}")
        # In a real scenario, we'd fall back to mock data or error out.
