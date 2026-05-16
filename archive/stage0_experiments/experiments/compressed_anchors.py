"""
experiments/compressed_anchors.py
Phase 13: Learned / Compressed Anchor Experiments
Investigates whether exact semantic stabilization can emerge WITHOUT full KV restoration.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

class CompressedAnchorExperiment:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def run_experiment(self, text: str, anchor_pos: int, noise_std: float = 0.2):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        context_len = input_ids.shape[1] - 1
        
        # 1. Baseline
        out_base = self.model(input_ids[:, :context_len], use_cache=True)
        kv_base = out_base.past_key_values
        if hasattr(kv_base, "to_legacy_cache"): kv_base = kv_base.to_legacy_cache()
        
        _, hidden_truth = self._run(input_ids[:, context_len:], kv_base)
        
        # 2. Variants of anchor compression
        ranks = [1, 2, 4, 8, 16, 32]
        results = {}
        
        # Exact baseline
        kv_exact = self._anchor(kv_base, noise_std, anchor_pos, mode="exact")
        _, h_exact = self._run(input_ids[:, context_len:], kv_exact)
        results["Exact"] = torch.norm(hidden_truth[-1].float() - h_exact[-1].float(), p=2).item()
        
        # No SAM baseline
        kv_noisy = self._add_noise(kv_base, noise_std)
        _, h_noisy = self._run(input_ids[:, context_len:], kv_noisy)
        results["No SAM"] = torch.norm(hidden_truth[-1].float() - h_noisy[-1].float(), p=2).item()
        
        # Low Rank Variants
        for r in ranks:
            kv_lr = self._anchor(kv_base, noise_std, anchor_pos, mode="low_rank", rank=r)
            _, h_lr = self._run(input_ids[:, context_len:], kv_lr)
            results[f"Low-Rank (R={r})"] = torch.norm(hidden_truth[-1].float() - h_lr[-1].float(), p=2).item()
            
        return results

    def _run(self, ids, kv):
        hidden_states = []
        def hook_fn(m, i, o):
            hidden_states.append(o[0].detach().cpu() if isinstance(o, tuple) else o.detach().cpu())
        hooks = [l.register_forward_hook(hook_fn) for l in self.model.model.layers]
        out = self.model(ids, past_key_values=DynamicCache.from_legacy_cache(kv))
        for h in hooks: h.remove()
        return out, hidden_states

    def _add_noise(self, kv, std):
        return tuple([(k + torch.randn_like(k) * std, v + torch.randn_like(v) * std) for k, v in kv])

    def _anchor(self, kv, std, anchor_pos, mode="exact", rank=8):
        noisy = self._add_noise(kv, std)
        anchored = []
        for l in range(self.num_layers):
            k, v = noisy[l][0].clone(), noisy[l][1].clone()
            
            if mode == "exact":
                k[:, :, anchor_pos, :] = kv[l][0][:, :, anchor_pos, :]
                v[:, :, anchor_pos, :] = kv[l][1][:, :, anchor_pos, :]
            elif mode == "low_rank":
                # Compress the key and value for this anchor using SVD
                # k: [b, h, s, d] -> we care about [h, d] for this pos
                kh = kv[l][0][0, :, anchor_pos, :] # [heads, dim]
                vh = kv[l][1][0, :, anchor_pos, :]
                
                # SVD of [heads, dim]
                # We want to approximate kh ≈ U S V^T
                # kh: [32, 128]
                U, S, Vh = torch.linalg.svd(kh.float(), full_matrices=False)
                kh_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).to(kh.dtype)
                
                U, S, Vh = torch.linalg.svd(vh.float(), full_matrices=False)
                vh_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).to(vh.dtype)
                
                k[:, :, anchor_pos, :] = kh_approx
                v[:, :, anchor_pos, :] = vh_approx
            
            anchored.append((k, v))
        return tuple(anchored)

    def visualize(self, results, save_path):
        names = list(results.keys())
        dists = list(results.values())
        
        plt.figure(figsize=(12, 6))
        plt.plot(names, dists, marker='o', linestyle='-', color='purple')
        plt.xticks(rotation=45)
        plt.ylabel("L2 Drift")
        plt.title("Scaling of Anchor Fidelity vs Manifold Stability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    exp = CompressedAnchorExperiment()
    text = "The scalability of semantic memory depends on the efficiency of its storage format. Low-rank approximations might provide a middle ground."
    results = exp.run_experiment(text, anchor_pos=10)
    
    os.makedirs("results/phase13/plots", exist_ok=True)
    exp.visualize(results, "results/phase13/plots/compressed_anchors_scaling.png")
    
    with open("results/phase13/compressed_anchors_results.json", "w") as f:
        json.dump(results, f, indent=4)
