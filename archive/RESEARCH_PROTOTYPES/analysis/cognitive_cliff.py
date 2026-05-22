"""
analysis/cognitive_cliff.py
Phase 14: Cognitive Cliff Analysis
Measures reasoning collapse dynamics across noise levels.
"""

import os
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from analysis.reasoning_manifold import ReasoningTrajectoryTracker

def run_cliff_analysis(tracker, prompts, noise_levels):
    results = []
    
    for noise in tqdm(noise_levels, desc="Noise Levels"):
        noise_results = {"noise": noise, "survival_rates": [], "drift_l2": []}
        
        for prompt in prompts:
            res = tracker.analyze_reasoning_stability(prompt, noise_std=noise) # No anchors by default in this loop
            
            # Survival rate = match tokens / total tokens
            matches = sum(1 for m in res["metrics_noise"] if m["token_match"])
            survival = matches / len(res["metrics_noise"]) if res["metrics_noise"] else 0
            
            # Final drift
            final_drift = res["metrics_noise"][-1]["layer_l2"][-1] if res["metrics_noise"] else 0
            
            noise_results["survival_rates"].append(survival)
            noise_results["drift_l2"].append(final_drift)
            
        results.append({
            "noise": noise,
            "avg_survival": sum(noise_results["survival_rates"]) / len(prompts),
            "avg_drift": sum(noise_results["drift_l2"]) / len(prompts)
        })
        
    return results

if __name__ == "__main__":
    tracker = ReasoningTrajectoryTracker()
    prompts = [
        "Question: If a train travels at 60 mph for 2 hours and then at 80 mph for 3 hours, what is the total distance traveled? Let's think step by step.",
        "Question: A box contains 3 red balls and 5 blue balls. If I pick one ball, what is the probability it is red? Let's solve it step by step.",
        "Question: If x + 2 = 5, what is x? Explain your reasoning."
    ]
    
    noises = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    cliff_results = run_cliff_analysis(tracker, prompts, noises)
    
    os.makedirs("results/phase14/plots", exist_ok=True)
    
    noises = [r["noise"] for r in cliff_results]
    survivals = [r["avg_survival"] for r in cliff_results]
    drifts = [r["avg_drift"] for r in cliff_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(noises, survivals, marker='o', label="Reasoning Survival Rate")
    plt.xlabel("Noise Level")
    plt.ylabel("Survival Rate (Token Match)")
    plt.title("Cognitive Cliff: Reasoning Collapse")
    plt.grid(True)
    plt.savefig("results/phase14/plots/cognitive_cliff.png")
    plt.close()
    
    with open("results/phase14/cognitive_cliff_results.json", "w") as f:
        json.dump(cliff_results, f, indent=4)
