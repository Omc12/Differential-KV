"""
experiments/long_horizon_recovery.py
Phase 18: Evolutionary Manifold Shaping
Tests recovery capacity after significant (20-50 token) destabilization.
"""

import torch
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
from anchor_logic.basin_reinforcement import BasinReinforcementSystem

class LongHorizonRecoveryExperiment:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.tracker = ReasoningTrajectoryTracker(model_id=model_id, device=device)
        self.reinforcer = BasinReinforcementSystem()

    def run_experiment(self, prompt: str, destabilize_len: int = 30, noise_std: float = 0.2):
        """
        1. Run baseline.
        2. Run with noise for 'destabilize_len' tokens.
        3. Switch to SAM/Reinforced recovery mode.
        4. Measure if it returns to the baseline manifold.
        """
        print(f"Running Long-Horizon Recovery (Destabilize: {destabilize_len} tokens)...")
        
        # 1. Baseline
        ids_base, traj_base = self.tracker.run_generation(prompt, max_new_tokens=100)
        
        # 2. Recovery Test
        def recovery_mod(l_idx, k, v):
            step_idx = len(self.tracker.captured_hidden_states) // self.tracker.num_layers
            
            # Destabilization phase
            if step_idx < destabilize_len:
                return k + torch.randn_like(k) * noise_std, v + torch.randn_like(v) * noise_std
            
            # Recovery phase: apply reinforcement
            # Here we mock the stability map as distance to base manifold (inverse)
            return self.reinforcer.apply_reinforcement(l_idx, k, v, torch.ones_like(k[:, :, 0, 0], dtype=k.dtype) * 0.9)

        ids_rec, traj_rec = self.tracker.run_generation(prompt, max_new_tokens=100, kv_modifier_fn=recovery_mod)
        
        # Measure divergence
        metrics = self.tracker.measure_divergence(traj_base, traj_rec)
        
        return {
            "traj_base": traj_base,
            "traj_rec": traj_rec,
            "metrics": metrics,
            "text_base": self.tracker.tokenizer.decode(ids_base[0]),
            "text_rec": self.tracker.tokenizer.decode(ids_rec[0])
        }

    def plot_recovery(self, results: Dict, save_path: str):
        l2_drifts = [m["layer_l2"][-1] for m in results["metrics"]]
        
        plt.figure(figsize=(10, 5))
        plt.plot(l2_drifts, label="L2 Drift (Last Layer)")
        plt.axvline(x=30, color='r', linestyle='--', label="Intervention Start")
        plt.title("Long-Horizon Cognitive Recovery")
        plt.xlabel("Tokens Generated")
        plt.ylabel("Divergence from Baseline")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    exp = LongHorizonRecoveryExperiment()
    prompt = "Explain the relationship between quantum entanglement and non-locality in detail."
    res = exp.run_experiment(prompt)
    
    os.makedirs("results/phase18/plots", exist_ok=True)
    exp.plot_recovery(res, "results/phase18/plots/long_horizon_recovery.png")
    
    with open("results/phase18/long_horizon_recovery.json", "w") as f:
        json.dump({
            "text_base": res["text_base"],
            "text_rec": res["text_rec"]
        }, f, indent=4)
