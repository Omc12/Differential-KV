"""
analysis/reasoning_manifold.py
Phase 14: Reasoning Manifold Preservation
Implements trajectory tracking, bifurcation analysis, and attention circuit mapping for reasoning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, Cache
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from tqdm import tqdm
from anchor_logic.semantic_anchor_system import AnchorSelectionPolicy
from anchor_logic.reasoning_anchors import ReasoningAnchor, ReasoningAnchorMemory, ChainOfThoughtPolicy

class ReasoningTrajectoryTracker:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        print(f"Loading model {model_id} for Reasoning Analysis...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
        # Hooks to capture hidden states and attentions
        self.captured_hidden_states = []
        self.captured_attentions = []
        self.hooks = []

    def _attach_hooks(self):
        self.captured_hidden_states = []
        self.captured_attentions = []
        self.hooks = []
        
        def hidden_hook(module, input, output):
            if isinstance(output, tuple):
                self.captured_hidden_states.append(output[0].detach().cpu())
            else:
                self.captured_hidden_states.append(output.detach().cpu())
                
        def attn_hook(module, input, output):
            # output is (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) > 1:
                if output[1] is not None:
                    self.captured_attentions.append(output[1].detach().cpu())

        for layer_idx, layer in enumerate(self.model.model.layers):
            self.hooks.append(layer.register_forward_hook(hidden_hook))
            # Hook the attention module specifically for weights
            self.hooks.append(layer.self_attn.register_forward_hook(attn_hook))
            
            # To capture KV, we can use the past_key_value from output of the layer
            # but it's cleaner to just get it from the model output.
            # However, for simulation, we need layer-wise access.

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def run_generation(self, prompt: str, max_new_tokens: int = 50, kv_modifier_fn=None):
        """
        Runs generation while capturing trajectories.
        kv_modifier_fn: A function that takes (layer_idx, k, v) and returns modified (k, v).
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids.clone()
        
        trajectories = [] # List of (hidden_states, attentions) per step
        past_key_values = None
        
        for i in range(max_new_tokens):
            self._attach_hooks()
            # If we have a modifier, we need to manually handle the cache
            if kv_modifier_fn and past_key_values:
                # Modify existing cache
                # past_key_values could be DynamicCache or tuple
                if isinstance(past_key_values, Cache):
                    legacy_kv = past_key_values.to_legacy_cache()
                else:
                    legacy_kv = past_key_values
                
                new_cache = []
                for l_idx, (k, v) in enumerate(legacy_kv):
                    mk, mv = kv_modifier_fn(l_idx, k, v)
                    new_cache.append((mk, mv))
                
                past_key_values = DynamicCache.from_legacy_cache(tuple(new_cache))

            outputs = self.model(
                input_ids=generated_ids[:, -1:] if i > 0 else generated_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True
            )
            self._remove_hooks()
            
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1:].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            trajectories.append({
                "hidden": [h.clone() for h in self.captured_hidden_states],
                "attn": [a.clone() for a in self.captured_attentions],
                "kv": [ (k.clone(), v.clone()) for (k, v) in outputs.past_key_values ],
                "token": next_token_id.item()
            })
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
        return generated_ids, trajectories

    def measure_divergence(self, base_traj: List[Dict], test_traj: List[Dict]):
        """
        Measures drift between two trajectories token-by-token.
        """
        drift_metrics = []
        min_len = min(len(base_traj), len(test_traj))
        
        for i in range(min_len):
            step_drift = {"layer_l2": [], "layer_cos": [], "token_match": False}
            
            bh = base_traj[i]["hidden"]
            th = test_traj[i]["hidden"]
            
            for l in range(len(bh)):
                b = bh[l][:, -1, :].float()
                t = th[l][:, -1, :].float()
                
                l2 = torch.norm(b - t, p=2).item()
                cos = F.cosine_similarity(b, t, dim=-1).item()
                
                step_drift["layer_l2"].append(l2)
                step_drift["layer_cos"].append(cos)
            
            step_drift["token_match"] = (base_traj[i]["token"] == test_traj[i]["token"])
            drift_metrics.append(step_drift)
            
        return drift_metrics

    def plot_trajectories(self, base_traj: List[Dict], noise_traj: List[Dict], sam_traj: List[Dict], title: str, save_path: str):
        """
        Visualizes trajectories in 2D using PCA on the final layer hidden states.
        """
        base_states = [t["hidden"][-1][:, -1, :].numpy().flatten() for t in base_traj]
        noise_states = [t["hidden"][-1][:, -1, :].numpy().flatten() for t in noise_traj]
        sam_states = [t["hidden"][-1][:, -1, :].numpy().flatten() for t in sam_traj]
        
        all_states = np.array(base_states + noise_states + sam_states)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_states)
        
        n_base = len(base_states)
        n_noise = len(noise_states)
        
        base_coords = coords[:n_base]
        noise_coords = coords[n_base:n_base+n_noise]
        sam_coords = coords[n_base+n_noise:]
        
        plt.figure(figsize=(10, 8))
        plt.plot(base_coords[:, 0], base_coords[:, 1], 'b-o', label="FP16 Baseline", alpha=0.6)
        plt.plot(noise_coords[:, 0], noise_coords[:, 1], 'r--x', label="Noisy (Collapsed)", alpha=0.6)
        plt.plot(sam_coords[:, 0], sam_coords[:, 1], 'g-.s', label="SAM (Stabilized)", alpha=0.6)
        
        # Mark start and end
        plt.scatter(base_coords[0, 0], base_coords[0, 1], c='green', s=100, marker='^', label="Start")
        plt.scatter(base_coords[-1, 0], base_coords[-1, 1], c='black', s=100, marker='x', label="End")
        
        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def analyze_attention_paths(self, traj: List[Dict]):
        """
        Maps the attention topology over time.
        Identifies 'Reasoning Circuits' (heads with high persistence).
        """
        num_layers = len(traj[0]["attn"])
        num_heads = traj[0]["attn"][0].shape[1]
        
        # persistence[layer, head] = variance of attention distribution over time
        # or entropy stability. Let's use mean entropy.
        circuit_stability = torch.zeros(num_layers, num_heads)
        
        for step in traj:
            for l in range(num_layers):
                attn = step["attn"][l] # [1, heads, q_len, k_len]
                # Last query's attention
                q_attn = attn[0, :, -1, :]
                entropy = -torch.sum(q_attn * torch.log(q_attn + 1e-9), dim=-1)
                circuit_stability[l] += entropy.cpu()
        
        circuit_stability /= len(traj)
        return circuit_stability

    def simulate_sam_generation(self, prompt: str, base_traj: List[Dict], noise_std: float = 0.1, anchor_rank: Optional[int] = None):
        """
        Simulates generation with SAM-based healing.
        anchor_rank: If provided, uses low-rank approximation for anchored states.
        """
        print(f"Simulating SAM-enabled generation (Low-Rank: {anchor_rank})...")
        
        cot_keywords = ["step", "therefore", "thus", "because", "so", "first", "finally", "python", "def", "answer"]
        all_tokens = [t["token"] for t in base_traj]
        anchored_positions = []
        for i, tid in enumerate(all_tokens):
            text = self.tokenizer.decode([tid]).lower()
            if any(k in text for k in cot_keywords):
                anchored_positions.append(i)
        
        if not anchored_positions:
            anchored_positions = list(range(0, len(all_tokens), 16))

        def low_rank_approx(tensor, rank):
            # tensor: [batch, heads, 1, dim] or [batch, heads, seq, dim]
            # We want to approximate the 'dim' dimension
            u, s, v = torch.svd(tensor.float())
            u_r = u[..., :rank]
            s_r = s[..., :rank]
            v_r = v[..., :rank]
            return (u_r @ torch.diag_embed(s_r) @ v_r.transpose(-1, -2)).to(tensor.dtype)

        def sam_mod(l_idx, k, v):
            k_noisy = k + torch.randn_like(k) * noise_std
            v_noisy = v + torch.randn_like(v) * noise_std
            
            seq_len = k.shape[2]
            for pos in anchored_positions:
                if pos < seq_len:
                    base_k, base_v = base_traj[-1]["kv"][l_idx]
                    bk = base_k[:, :, pos:pos+1, :]
                    bv = base_v[:, :, pos:pos+1, :]
                    
                    if anchor_rank:
                        bk = low_rank_approx(bk, anchor_rank)
                        bv = low_rank_approx(bv, anchor_rank)
                        
                    k_noisy[:, :, pos:pos+1, :] = bk
                    v_noisy[:, :, pos:pos+1, :] = bv
            
            return k_noisy, v_noisy
            
        return self.run_generation(prompt, kv_modifier_fn=sam_mod)

    def analyze_reasoning_stability(self, prompt: str, noise_std: float = 0.05):
        """
        Compare FP16 vs Noisy vs SAM.
        """
        print(f"\nAnalyzing Stability for: {prompt[:50]}...")
        
        # 1. Baseline
        ids_base, traj_base = self.run_generation(prompt)
        text_base = self.tokenizer.decode(ids_base[0])
        
        # 2. Noisy KV
        def noise_mod(l, k, v):
            return k + torch.randn_like(k) * noise_std, v + torch.randn_like(v) * noise_std
            
        ids_noise, traj_noise = self.run_generation(prompt, kv_modifier_fn=noise_mod)
        text_noise = self.tokenizer.decode(ids_noise[0])
        
        # 3. SAM Enabled (Using CoT policy + Low-Rank)
        ids_sam, traj_sam = self.simulate_sam_generation(prompt, traj_base, noise_std=noise_std, anchor_rank=8)
        text_sam = self.tokenizer.decode(ids_sam[0])
        
        metrics_noise = self.measure_divergence(traj_base, traj_noise)
        metrics_sam = self.measure_divergence(traj_base, traj_sam)
        
        return {
            "text_base": text_base,
            "text_noise": text_noise,
            "text_sam": text_sam,
            "metrics_noise": metrics_noise,
            "metrics_sam": metrics_sam,
            "traj_base": traj_base,
            "traj_noise": traj_noise,
            "traj_sam": traj_sam
        }

if __name__ == "__main__":
    tracker = ReasoningTrajectoryTracker()
    prompt = "Question: If a train travels at 60 mph for 2 hours and then at 80 mph for 3 hours, what is the total distance traveled? Let's think step by step."
    
    res = tracker.analyze_reasoning_stability(prompt, noise_std=0.1)
    
    os.makedirs("results/phase14/plots", exist_ok=True)
    tracker.plot_trajectories(res["traj_base"], res["traj_noise"], res["traj_sam"], "Reasoning Trajectory Stabilizaton", "results/phase14/plots/reasoning_stabilization.png")
    
    print(f"\n--- RESULTS ---")
    print(f"Base Output: {res['text_base'].encode('ascii', 'ignore').decode('ascii')}")
    print(f"Noise Output: {res['text_noise'].encode('ascii', 'ignore').decode('ascii')}")
    print(f"SAM Output: {res['text_sam'].encode('ascii', 'ignore').decode('ascii')}")
    
    # Task 4: Attention Path Analysis
    stability_base = tracker.analyze_attention_paths(res["traj_base"])
    stability_noise = tracker.analyze_attention_paths(res["traj_noise"])
    stability_sam = tracker.analyze_attention_paths(res["traj_sam"])
    
    # Visualize circuit stability
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(stability_base); plt.title("Base Circuit Stability")
    plt.subplot(1, 3, 2); plt.imshow(stability_noise); plt.title("Noise Circuit Stability")
    plt.subplot(1, 3, 3); plt.imshow(stability_sam); plt.title("SAM Circuit Stability")
    plt.savefig("results/phase14/plots/circuit_stability.png")
    
    with open("results/phase14/reasoning_trajectory_results.json", "w") as f:
        json.dump({
            "prompt": prompt,
            "text_base": res["text_base"],
            "text_noise": res["text_noise"],
            "text_sam": res["text_sam"],
            "metrics_noise": res["metrics_noise"],
            "metrics_sam": res["metrics_sam"]
        }, f, indent=4)
