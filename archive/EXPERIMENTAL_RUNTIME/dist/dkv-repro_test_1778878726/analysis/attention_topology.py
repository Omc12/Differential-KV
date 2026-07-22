"""
analysis/attention_topology.py
Phase 13: Attention Topology Analysis
Analyzes attention behavior around anchors and compares topology across variants.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json

class AttentionTopologyAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="eager").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

    def _prepare_kv(self, kv):
        """Ensures KV is in a format the model accepts (tuple of tuples)."""
        if hasattr(kv, "to_legacy_cache"):
            return kv.to_legacy_cache()
        return tuple(kv)

    @torch.no_grad()
    def analyze_topology(self, text: str, anchor_pos: int, noise_std: float = 0.1):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        context_ids = input_ids[:, :-1]
        query_ids = input_ids[:, -1:]
        
        # 1. FP16 Baseline
        outputs_ctx = self.model(context_ids, use_cache=True)
        kv_base_raw = outputs_ctx.past_key_values
        kv_base = self._prepare_kv(kv_base_raw)
        
        # Baseline Attention for Query
        outputs_base = self.model(query_ids, past_key_values=DynamicCache.from_legacy_cache(kv_base), output_attentions=True)
        attn_base = outputs_base.attentions
            
        # 2. Noisy KV
        kv_noisy = []
        for l in range(self.num_layers):
            k, v = kv_base[l][0], kv_base[l][1]
            kv_noisy.append((k + torch.randn_like(k) * noise_std, v + torch.randn_like(v) * noise_std))
        
        # Run with noisy KV
        outputs_noisy = self.model(query_ids, past_key_values=DynamicCache.from_legacy_cache(tuple(kv_noisy)), output_attentions=True)
        attn_noisy = outputs_noisy.attentions
        
        # 3. SAM Enabled (Anchored)
        kv_sam = []
        for l in range(self.num_layers):
            k, v = kv_noisy[l][0].clone(), kv_noisy[l][1].clone()
            k[:, :, anchor_pos, :] = kv_base[l][0][:, :, anchor_pos, :]
            v[:, :, anchor_pos, :] = kv_base[l][1][:, :, anchor_pos, :]
            kv_sam.append((k, v))
            
        outputs_sam = self.model(query_ids, past_key_values=DynamicCache.from_legacy_cache(tuple(kv_sam)), output_attentions=True)
        attn_sam = outputs_sam.attentions
        
        # Comparison Metrics
        # Focus on the query token's attention weights
        results = []
        for l in range(self.num_layers):
            # All attn should now be [1, heads, 1, context_len + 1]
            ab = attn_base[l][0, :, 0, :].cpu().float().numpy()
            an = attn_noisy[l][0, :, 0, :].cpu().float().numpy()
            as_ = attn_sam[l][0, :, 0, :].cpu().float().numpy()
            
            # Entropy
            ent_b = -np.sum(ab * np.log(ab + 1e-9), axis=-1).mean()
            ent_n = -np.sum(an * np.log(an + 1e-9), axis=-1).mean()
            ent_s = -np.sum(as_ * np.log(as_ + 1e-9), axis=-1).mean()
            
            # Divergence from baseline
            # JS divergence or KL? Let's use simple MSE for weights for now
            mse_n = np.mean((ab - an)**2)
            mse_s = np.mean((ab - as_)**2)
            
            results.append({
                "layer": l,
                "entropy": {"base": float(ent_b), "noisy": float(ent_n), "sam": float(ent_s)},
                "mse_to_base": {"noisy": float(mse_n), "sam": float(mse_s)}
            })
            
        return results

    def visualize_topology(self, results, save_path):
        layers = [r["layer"] for r in results]
        ent_b = [r["entropy"]["base"] for r in results]
        ent_n = [r["entropy"]["noisy"] for r in results]
        ent_s = [r["entropy"]["sam"] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, ent_b, label="Baseline (FP16)", marker='o')
        plt.plot(layers, ent_n, label="Compressed (Noisy)", marker='x')
        plt.plot(layers, ent_s, label="SAM (Anchored)", marker='s')
        plt.xlabel("Layer")
        plt.ylabel("Attention Entropy")
        plt.title("Attention Topology Stabilization")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Topology visualization saved to {save_path}")

if __name__ == "__main__":
    analyzer = AttentionTopologyAnalyzer()
    text = "The secret of understanding transformer mechanics lies in the topology of its attention heads. Anchors restore these pathways."
    results = analyzer.analyze_topology(text, anchor_pos=5)
    
    os.makedirs("results/phase13/plots", exist_ok=True)
    analyzer.visualize_topology(results, "results/phase13/plots/attention_topology.png")
    
    with open("results/phase13/attention_topology_results.json", "w") as f:
        json.dump(results, f, indent=4)
