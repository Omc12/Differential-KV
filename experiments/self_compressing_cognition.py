"""
experiments/self_compressing_cognition.py
Phase 19: Universal Cognitive Geometry
Tests whether models can self-organize sparse cognition and maintain stable compressed trajectories.
"""

import torch
import numpy as np
import os
import json
from analysis.reasoning_manifold import ReasoningTrajectoryTracker
from typing import List, Dict, Any, Tuple

class SelfCompressingCognitionExperiment:
    def __init__(self, model_id: str = "Qwen/Qwen2-0.5B", device="cuda"):
        self.tracker = ReasoningTrajectoryTracker(model_id=model_id, device=device)
        self.device = device

    def run_self_compression_test(self, prompt: str, target_sparsity: float = 0.5):
        """
        Runs a test where the model's KV cache is progressively pruned/compressed
        based on the model's own 'certainty' or importance metrics.
        """
        print(f"Running Self-Compression Test for {self.tracker.model.config._name_or_path}...")
        
        # 1. Baseline Run
        ids_base, traj_base = self.tracker.run_generation(prompt)
        
        # 2. Self-Compressed Run
        # We simulate 'self-compression' by using the model's own attention entropy
        # to decide which KV pairs to keep.
        
        def self_compress_mod(l_idx, k, v):
            # Use attention from the last step if available
            # For simplicity in this experiment, we prune the least-attended tokens
            # based on the current layer's attention distribution.
            
            # k: [batch, heads, seq, dim]
            seq_len = k.shape[2]
            if seq_len < 10: return k, v
            
            # Simple heuristic: keep the first 4 tokens (anchors) and the last 4 tokens (context)
            # and prune the middle if they have low 'importance'.
            # Real self-compression would use a learned policy.
            
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=k.device)
            num_to_prune = int(seq_len * target_sparsity)
            
            if num_to_prune > 0:
                # Mock importance: distance from centroid
                centroid = k.mean(dim=2, keepdim=True)
                importance = torch.norm(k - centroid, dim=-1).mean(dim=1).squeeze(0) # [seq]
                
                _, indices = torch.topk(importance, k=seq_len - num_to_prune, largest=True)
                keep_mask[:] = False
                keep_mask[indices] = True
                # Always keep BOS and last tokens
                keep_mask[0] = True
                keep_mask[-1] = True
                
            return k[:, :, keep_mask, :], v[:, :, keep_mask, :]

        # Note: run_generation needs to be slightly modified to handle varying seq_len in cache
        # or we just use the modifier to zero out or sparsify. 
        # For this prototype, we'll use 'simulated sparsity' (noise injection on low-importance).
        
        def simulated_sparsity_mod(l_idx, k, v):
            # Inject noise into low-importance states to simulate loss of precision
            centroid = k.mean(dim=2, keepdim=True)
            dist = torch.norm(k - centroid, dim=-1).mean(dim=1).squeeze(0)
            
            # Add noise to the bottom 50%
            threshold = torch.median(dist)
            noise_mask = dist < threshold
            
            k_mod = k.clone()
            v_mod = v.clone()
            
            # Mask needs to be broadcastable to [1, heads, seq, dim]
            mask_4d = noise_mask.view(1, 1, -1, 1)
            k_mod[mask_4d.expand_as(k)] += torch.randn_like(k[mask_4d.expand_as(k)]) * 0.2
            v_mod[mask_4d.expand_as(v)] += torch.randn_like(v[mask_4d.expand_as(v)]) * 0.2
            
            return k_mod, v_mod

        ids_comp, traj_comp = self.tracker.run_generation(prompt, kv_modifier_fn=simulated_sparsity_mod)
        
        text_base = self.tracker.tokenizer.decode(ids_base[0])
        text_comp = self.tracker.tokenizer.decode(ids_comp[0])
        
        results = {
            "text_base": text_base,
            "text_comp": text_comp,
            "coherence_maintained": bool(text_base[:50] == text_comp[:50]) # Simple proxy
        }
        
        return results

if __name__ == "__main__":
    exp = SelfCompressingCognitionExperiment()
    prompt = "Step by step, explain how to calculate the area of a circle with radius 5."
    res = exp.run_self_compression_test(prompt)
    print("Self-Compression Results:", res)
    
    os.makedirs("results/phase19", exist_ok=True)
    with open("results/phase19/self_compression_analysis.json", "w") as f:
        json.dump(res, f, indent=4)
