"""
analysis/influence_analyzer.py
Phase 13: Anchor Influence Propagation Analysis
Measures the spatial and temporal spread of anchor stabilization effects.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

class AnchorInfluenceAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    def _prepare_kv(self, kv):
        """Ensures KV is in a format the model accepts (tuple of tuples)."""
        if hasattr(kv, "to_legacy_cache"):
            return kv.to_legacy_cache()
        return tuple(kv)

    @torch.no_grad()
    def measure_propagation(self, text: str, anchor_pos: int, noise_std: float = 0.1):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        
        # 1. FP16 Baseline
        outputs_base = self.model(input_ids, use_cache=True)
        kv_base = self._prepare_kv(outputs_base.past_key_values)
        
        hidden_base = self._capture_all_hidden(input_ids)
        
        # 2. No-SAM (Noisy) Pass
        kv_noisy = []
        for l in range(self.num_layers):
            k, v = kv_base[l][0], kv_base[l][1]
            k_noisy = k + torch.randn_like(k) * noise_std
            v_noisy = v + torch.randn_like(v) * noise_std
            kv_noisy.append((k_noisy, v_noisy))
        
        kv_noisy = tuple(kv_noisy)
        
        # 3. SAM (Anchored at anchor_pos)
        kv_sam = []
        for l in range(self.num_layers):
            k, v = kv_noisy[l][0].clone(), kv_noisy[l][1].clone()
            k[:, :, anchor_pos, :] = kv_base[l][0][:, :, anchor_pos, :]
            v[:, :, anchor_pos, :] = kv_base[l][1][:, :, anchor_pos, :]
            kv_sam.append((k, v))
            
        kv_sam = tuple(kv_sam)
            
        return self._calculate_influence_field(input_ids, kv_base, kv_noisy, kv_sam, anchor_pos)

    def _capture_all_hidden(self, input_ids, past_key_values=None):
        hidden_states = []
        def hook_fn(module, input, output):
            hidden_states.append(output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu())
        
        hooks = []
        for layer in self.model.model.layers:
            hooks.append(layer.register_forward_hook(hook_fn))
            
        self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        
        for h in hooks: h.remove()
        return hidden_states # List of [1, seq_len, hidden_size]

    def _calculate_influence_field(self, input_ids, kv_base, kv_noisy, kv_sam, anchor_pos):
        # We need to run the model in a way that it actually uses the pre-computed KV
        # But transformers .forward(input_ids, past_key_values) uses the KV for PREVIOUS tokens
        # and computes KV for CURRENT tokens.
        
        # So we want to run the model for tokens [anchor_pos+1 : ] using the modified KV[ : anchor_pos+1]
        
        remaining_ids = input_ids[:, anchor_pos+1:]
        
        # Baseline hidden states for remaining tokens
        # (Already have them from _capture_all_hidden on full sequence, just slice)
        
        # Hidden states for noisy KV
        kv_noisy_subset = tuple([(k[:, :, :anchor_pos+1, :], v[:, :, :anchor_pos+1, :]) for k, v in kv_noisy])
        hidden_noisy = self._capture_all_hidden(remaining_ids, past_key_values=DynamicCache.from_legacy_cache(kv_noisy_subset))
        
        # Hidden states for SAM KV
        kv_sam_subset = tuple([(k[:, :, :anchor_pos+1, :], v[:, :, :anchor_pos+1, :]) for k, v in kv_sam])
        hidden_sam = self._capture_all_hidden(remaining_ids, past_key_values=DynamicCache.from_legacy_cache(kv_sam_subset))
        
        # Baseline for comparison
        kv_base_subset = tuple([(k[:, :, :anchor_pos+1, :], v[:, :, :anchor_pos+1, :]) for k, v in kv_base])
        hidden_base = self._capture_all_hidden(remaining_ids, past_key_values=DynamicCache.from_legacy_cache(kv_base_subset))
        
        # Now compare hidden_noisy and hidden_sam against hidden_base
        # hidden_base: List of [1, remain_len, hidden_size]
        
        influence_map = [] # [layer, pos_offset]
        for l in range(self.num_layers):
            b = hidden_base[l].squeeze(0).float()
            n = hidden_noisy[l].squeeze(0).float()
            s = hidden_sam[l].squeeze(0).float()
            
            err_noisy = torch.norm(b - n, p=2, dim=-1)
            err_sam = torch.norm(b - s, p=2, dim=-1)
            
            # Stabilization = Improvement in error
            stabilization = (err_noisy - err_sam).cpu().numpy()
            influence_map.append(stabilization.tolist())
            
        return influence_map

    def visualize_influence(self, influence_map, save_path):
        influence_map = np.array(influence_map)
        plt.figure(figsize=(10, 6))
        plt.imshow(influence_map, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label="Stabilization Magnitude")
        plt.xlabel("Tokens after Anchor")
        plt.ylabel("Layer")
        plt.title("Anchor Influence Propagation Field")
        plt.savefig(save_path)
        plt.close()
        print(f"Influence visualization saved to {save_path}")

if __name__ == "__main__":
    analyzer = AnchorInfluenceAnalyzer()
    text = "The quick brown fox jumps over the lazy dog. " * 20 # Long text
    anchor_pos = 50
    influence = analyzer.measure_propagation(text, anchor_pos=anchor_pos)
    
    os.makedirs("results/phase13/plots", exist_ok=True)
    analyzer.visualize_influence(influence, "results/phase13/plots/influence_propagation.png")
    
    with open("results/phase13/anchor_influence_results.json", "w") as f:
        json.dump(influence, f)
