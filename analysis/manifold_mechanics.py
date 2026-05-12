"""
analysis/manifold_mechanics.py
Phase 13: Semantic Manifold Mechanics Analysis
Implements trajectory tracking, drift measurement, and phase transition analysis.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from tqdm import tqdm

class SemanticManifoldAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        print(f"Loading model {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
        # Hooks to capture hidden states
        self.captured_hidden_states = []
        self.hooks = []

    def _attach_hooks(self):
        self.captured_hidden_states = []
        self.hooks = []
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.captured_hidden_states.append(output[0].detach().cpu())
            else:
                self.captured_hidden_states.append(output.detach().cpu())
        
        for layer in self.model.model.layers:
            self.hooks.append(layer.register_forward_hook(hook_fn))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def run_inference(self, input_ids: torch.Tensor, past_key_values=None):
        self._attach_hooks()
        outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        self._remove_hooks()
        # hidden_states will have num_layers tensors, each [1, seq_len, hidden_size]
        return outputs, self.captured_hidden_states

    def get_drift_metrics(self, base_hidden, test_hidden):
        """
        Computes layer-wise distances between hidden states.
        base_hidden/test_hidden: List of [1, 1, hidden_size] tensors (for single token prediction)
        """
        cosine_sims = []
        l2_distances = []
        
        for b, t in zip(base_hidden, test_hidden):
            b = b.squeeze().float()
            t = t.squeeze().float()
            
            cos = F.cosine_similarity(b.unsqueeze(0), t.unsqueeze(0), dim=-1)
            cosine_sims.append(cos.item())
            
            l2 = torch.norm(b - t, p=2)
            l2_distances.append(l2.item())
            
        return {
            "cosine_sims": cosine_sims,
            "l2_distances": l2_distances
        }

    def _prepare_kv(self, kv):
        """Ensures KV is in a format the model accepts (tuple of tuples)."""
        if hasattr(kv, "to_legacy_cache"):
            return kv.to_legacy_cache()
        return tuple(kv)

    @torch.no_grad()
    def analyze_mechanics(self, text: str, anchor_interval: int = 64, noise_std: float = 0.05):
        """
        Main mechanistic analysis loop.
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        context_len = input_ids.shape[1] - 1
        
        # 1. FP16 Baseline Pass
        print(f"Running Baseline Pass (Context: {context_len})...")
        outputs_base, _ = self.run_inference(input_ids[:, :context_len])
        kv_base_raw = outputs_base.past_key_values
        kv_base = self._prepare_kv(kv_base_raw)
        
        # Capture next-token hidden states (The "Truth")
        _, hidden_fp16 = self.run_inference(input_ids[:, context_len:], past_key_values=DynamicCache.from_legacy_cache(kv_base))
        
        # 2. Compressed Pass (Simulate 8-bit or Low-Rank drift)
        print("Running Compressed Pass...")
        kv_comp = []
        for l in range(self.num_layers):
            k, v = kv_base[l][0], kv_base[l][1]
            # Add noise to simulate compression error
            k_err = k + torch.randn_like(k) * noise_std
            v_err = v + torch.randn_like(v) * noise_std
            kv_comp.append((k_err, v_err))
            
        kv_comp = tuple(kv_comp)
        _, hidden_comp = self.run_inference(input_ids[:, context_len:], past_key_values=DynamicCache.from_legacy_cache(kv_comp))
        
        # 3. SAM-enabled Pass (Compressed + Exact Anchors)
        print(f"Running SAM Pass (Interval: {anchor_interval})...")
        kv_sam = []
        anchored_positions = list(range(0, context_len, anchor_interval))
        
        for l in range(self.num_layers):
            k, v = kv_comp[l][0].clone(), kv_comp[l][1].clone()
            # Reinject exact values at anchor positions
            for pos in anchored_positions:
                k[:, :, pos, :] = kv_base[l][0][:, :, pos, :]
                v[:, :, pos, :] = kv_base[l][1][:, :, pos, :]
            kv_sam.append((k, v))
            
        kv_sam = tuple(kv_sam)
        _, hidden_sam = self.run_inference(input_ids[:, context_len:], past_key_values=DynamicCache.from_legacy_cache(kv_sam))
        
        # 4. Metrics Calculation
        metrics_comp = self.get_drift_metrics(hidden_fp16, hidden_comp)
        metrics_sam = self.get_drift_metrics(hidden_fp16, hidden_sam)
        
        return {
            "comp": metrics_comp,
            "sam": metrics_sam,
            "anchors": anchored_positions
        }

    def visualize_drift(self, results: Dict, save_path: str):
        layers = list(range(self.num_layers))
        
        plt.figure(figsize=(12, 5))
        
        # Plot Cosine Similarity
        plt.subplot(1, 2, 1)
        plt.plot(layers, results["comp"]["cosine_sims"], label="Compressed (Noise)", marker='o')
        plt.plot(layers, results["sam"]["cosine_sims"], label="SAM (Anchored)", marker='s')
        plt.xlabel("Layer")
        plt.ylabel("Cosine Similarity to FP16")
        plt.title("Trajectory Alignment")
        plt.legend()
        plt.grid(True)
        
        # Plot L2 Distance
        plt.subplot(1, 2, 2)
        plt.plot(layers, results["comp"]["l2_distances"], label="Compressed", marker='o')
        plt.plot(layers, results["sam"]["l2_distances"], label="SAM", marker='s')
        plt.xlabel("Layer")
        plt.ylabel("L2 Distance to FP16")
        plt.title("Manifold Drift")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Drift visualization saved to {save_path}")

    def analyze_phase_transition(self, text: str, noise_levels: List[float]):
        """
        Task 2: Investigate whether semantic cliffs are true phase transitions.
        """
        print("\n>>> Analyzing Phase Transitions...")
        results = []
        for noise in tqdm(noise_levels, desc="Noise Levels"):
            res = self.analyze_mechanics(text, noise_std=noise)
            # Track the 'final layer' distance as a proxy for semantic collapse
            results.append({
                "noise": noise,
                "comp_dist": res["comp"]["l2_distances"][-1],
                "sam_dist": res["sam"]["l2_distances"][-1]
            })
        return results

    def visualize_phase_transition(self, results: List[Dict], save_path: str):
        noises = [r["noise"] for r in results]
        comp_dists = [r["comp_dist"] for r in results]
        sam_dists = [r["sam_dist"] for r in results]
        
        plt.figure(figsize=(8, 6))
        plt.plot(noises, comp_dists, label="Compressed (No Anchors)", marker='o')
        plt.plot(noises, sam_dists, label="SAM (Anchored)", marker='s')
        plt.xlabel("Noise Level (Compression Error)")
        plt.ylabel("Final Layer L2 Drift")
        plt.title("Semantic Cliff Phase Transition")
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(save_path)
        plt.close()
        print(f"Phase transition visualization saved to {save_path}")

if __name__ == "__main__":
    analyzer = SemanticManifoldAnalyzer()
    text = "The emergence of intelligence in large language models is often attributed to the stabilization of high-dimensional semantic manifolds. Semantic Anchor Memory aims to preserve these manifolds using sparse exact memory points."
    
    # Task 1
    res = analyzer.analyze_mechanics(text, anchor_interval=32)
    os.makedirs("results/phase13/plots", exist_ok=True)
    analyzer.visualize_drift(res, "results/phase13/plots/manifold_drift.png")
    
    # Task 2
    noises = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    phase_res = analyzer.analyze_phase_transition(text, noises)
    analyzer.visualize_phase_transition(phase_res, "results/phase13/plots/phase_transition.png")
    
    with open("results/phase13/manifold_mechanics_results.json", "w") as f:
        json.dump({"drift": res, "phase_transition": phase_res}, f, indent=4)
