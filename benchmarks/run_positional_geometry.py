"""
benchmarks/run_positional_geometry.py
Phase 13: Positional Geometry Investigation
Tests if semantic stabilization depends on exact identity, positional geometry, or both.
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

class PositionalGeometryTester:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def run_test(self, text: str, noise_std: float = 0.1):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        context_len = input_ids.shape[1] - 1
        
        # Baseline
        out_base = self.model(input_ids[:, :context_len], use_cache=True)
        kv_base = out_base.past_key_values
        if hasattr(kv_base, "to_legacy_cache"): kv_base = kv_base.to_legacy_cache()
        
        # Truth hidden states
        _, hidden_truth = self._run(input_ids[:, context_len:], kv_base)
        
        # Variants
        variants = {
            "FP16": kv_base,
            "Compressed (Noise)": self._add_noise(kv_base, noise_std),
            "Exact Anchor": self._anchor(kv_base, noise_std, anchor_pos=10, mode="exact"),
            "Position-only Anchor": self._anchor(kv_base, noise_std, anchor_pos=10, mode="position_only"),
            "Shifted Anchor (+5)": self._anchor(kv_base, noise_std, anchor_pos=10, mode="shifted", shift=5),
            "Delayed Anchor (+10)": self._anchor(kv_base, noise_std, anchor_pos=10, mode="delayed", delay=10)
        }
        
        results = {}
        for name, kv in variants.items():
            _, hidden = self._run(input_ids[:, context_len:], kv)
            dist = torch.norm(hidden_truth[-1].float() - hidden[-1].float(), p=2).item()
            results[name] = dist
            
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

    def _anchor(self, kv, std, anchor_pos, mode="exact", shift=0, delay=0):
        noisy = self._add_noise(kv, std)
        anchored = []
        for l in range(self.num_layers):
            k, v = noisy[l][0].clone(), noisy[l][1].clone()
            
            if mode == "exact":
                k[:, :, anchor_pos, :] = kv[l][0][:, :, anchor_pos, :]
                v[:, :, anchor_pos, :] = kv[l][1][:, :, anchor_pos, :]
            elif mode == "position_only":
                # Only restore the mean/norm or some positional signal?
                # For this test, we'll restore KV from a DIFFERENT token to simulate 'position exists but identity is wrong'
                wrong_pos = (anchor_pos + 5) % k.shape[2]
                k[:, :, anchor_pos, :] = kv[l][0][:, :, wrong_pos, :]
                v[:, :, anchor_pos, :] = kv[l][1][:, :, wrong_pos, :]
            elif mode == "shifted":
                # Restore exact KV but at a SHIFTED position in the cache
                target_pos = anchor_pos + shift
                if target_pos < k.shape[2]:
                    k[:, :, target_pos, :] = kv[l][0][:, :, anchor_pos, :]
                    v[:, :, target_pos, :] = kv[l][1][:, :, anchor_pos, :]
            
            anchored.append((k, v))
        return tuple(anchored)

    def visualize(self, results, save_path):
        names = list(results.keys())
        dists = list(results.values())
        
        plt.figure(figsize=(12, 6))
        plt.barh(names, dists, color='skyblue')
        plt.xlabel("L2 Drift (Final Layer)")
        plt.title("Positional Geometry vs Semantic Identity")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Positional geometry visualization saved to {save_path}")

if __name__ == "__main__":
    tester = PositionalGeometryTester()
    text = "The geometry of space-time is linked to the distribution of matter. Similarly, the geometry of latent space is linked to semantic anchors."
    results = tester.run_test(text)
    
    os.makedirs("results/phase13/plots", exist_ok=True)
    tester.visualize(results, "results/phase13/plots/positional_geometry.png")
    
    with open("results/phase13/positional_geometry_results.json", "w") as f:
        json.dump(results, f, indent=4)
