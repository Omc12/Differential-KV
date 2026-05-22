"""
evaluation/phase9_semantic_mapping.py
Phase 9: Semantic Importance Mapping for Differential KV.
Identifies critical heads, layers, and tokens responsible for model intelligence.
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics_utils import stable_kl_divergence, stable_attention_entropy, verify_retrieval
from compression.shared_basis import SharedBasisManager
from compression.adaptive import AdaptiveRankSelector
from compression.quantization import quantize_int8, dequantize_int8
from evaluation.perplexity_eval import Phase8PerplexityEvaluator
from evaluation.needle_haystack import NeedleHaystackEvaluator
from evaluation.generation_eval import GenerationEvaluator

class SemanticImportanceAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.ev = Phase8PerplexityEvaluator(model_id=model_id, device=device)
        self.model = self.ev.model
        self.tokenizer = self.ev.tokenizer
        self.device = self.ev.device
        self.needle_ev = NeedleHaystackEvaluator(self.ev)
        self.gen_ev = GenerationEvaluator(self.ev)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads

    def compress_kv_selective(
        self, 
        past_kv: Any, 
        mode: str, 
        layer_mask: Optional[Set[int]] = None, 
        head_mask: Optional[Dict[int, Set[int]]] = None,
        interval: int = 64
    ) -> Any:
        """
        Compresses KV cache selectively.
        layer_mask: Set of layer indices to compress. If None, all layers compressed.
        head_mask: Map of layer_idx -> set of head_indices to compress in that layer. 
                   If None, all heads in the layer are compressed.
        """
        is_cache_obj = hasattr(past_kv, "get_seq_length")
        kv_list = list(past_kv) if is_cache_obj else past_kv
        
        recon_kv_list = []
        
        for layer_idx, layer_data in enumerate(kv_list):
            # Check if this layer should be compressed at all
            if layer_mask is not None and layer_idx not in layer_mask:
                # Keep as FP16
                recon_kv_list.append(layer_data)
                continue
                
            k, v = layer_data[0], layer_data[1]
            b, h, s, d = k.shape
            
            # Check if specific heads should be compressed
            if head_mask is not None and layer_idx in head_mask:
                heads_to_compress = head_mask[layer_idx]
                
                k_recon = k.clone()
                v_recon = v.clone()
                
                for hi in heads_to_compress:
                    # Compress only head hi
                    # Extract head data
                    kh = k[:, hi:hi+1, :, :] # [b, 1, s, d]
                    vh = v[:, hi:hi+1, :, :]
                    
                    # Convert to flat format for Phase 8 compression logic
                    # We can reuse compress_kv by passing a dummy past_kv with just this head
                    dummy_kv = [(kh, vh)]
                    recon_dummy, _ = self.ev.compress_kv(dummy_kv, mode, interval=interval)
                    
                    k_recon[:, hi:hi+1, :, :] = recon_dummy[0][0]
                    v_recon[:, hi:hi+1, :, :] = recon_dummy[0][1]
                
                recon_kv_list.append((k_recon, v_recon) + layer_data[2:])
            else:
                # Compress entire layer
                dummy_kv = [(k, v)]
                recon_dummy, _ = self.ev.compress_kv(dummy_kv, mode, interval=interval)
                recon_kv_list.append(recon_dummy[0] + layer_data[2:])
                
        if is_cache_obj:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for i, layer_data in enumerate(recon_kv_list):
                new_cache.update(layer_data[0], layer_data[1], layer_idx=i)
            return new_cache
        else:
            return tuple(recon_kv_list)

    @torch.no_grad()
    def evaluate_sensitivity(self, mode="Layer-Shared Rank8", samples=2, context_len=1024):
        """
        Runs Task 1 & 2: Head-wise and Layer-wise sensitivity analysis.
        """
        # 1. Layer-wise sensitivity
        layer_results = []
        print("\n>>> Running Layer-wise Sensitivity Analysis...")
        for l in tqdm(range(self.num_layers), desc="Layers"):
            # Compress ONLY layer l
            # We use a helper to get a single sample metrics
            metrics = self._get_quick_metrics(mode, layer_mask={l}, samples=samples, context_len=context_len)
            metrics["layer"] = l
            layer_results.append(metrics)
            
        # 2. Head-wise sensitivity (sampled to avoid massive run time)
        # We'll pick some layers (e.g. first, middle, last) or all if small
        head_results = []
        print("\n>>> Running Head-wise Sensitivity Analysis...")
        # To be scientific but efficient, let's do all heads but with very few samples
        # Or just specific layers. Let's try all heads for this 0.5B model.
        for l in range(self.num_layers):
            for h in tqdm(range(self.num_heads), desc=f"Layer {l} Heads", leave=False):
                metrics = self._get_quick_metrics(mode, layer_mask={l}, head_mask={l: {h}}, samples=1, context_len=512)
                metrics["layer"] = l
                metrics["head"] = h
                head_results.append(metrics)
                
        return layer_results, head_results

    def _get_quick_metrics(self, mode, layer_mask=None, head_mask=None, samples=1, context_len=512):
        """
        Calculates KL, Top-K, and Retrieval for a specific compression mask.
        """
        # Use a fixed prompt for consistency
        prompt = "The quick brown fox jumps over the lazy dog. Scientific research is the systematic investigation into and study of materials and sources in order to establish facts and reach new conclusions."
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Baseline
        outputs_base = self.model(input_ids[:, :-1], use_cache=True)
        past_kv_base = outputs_base.past_key_values
        logits_base = self.model(input_ids[:, -1:], past_key_values=past_kv_base).logits[:, -1, :]
        log_probs_base = F.log_softmax(logits_base, dim=-1)
        
        # Compressed
        past_kv_comp = self.compress_kv_selective(past_kv_base, mode, layer_mask=layer_mask, head_mask=head_mask)
        logits_comp = self.model(input_ids[:, -1:], past_key_values=past_kv_comp).logits[:, -1, :]
        probs_comp = F.softmax(logits_comp, dim=-1)
        
        # Use stable KL implementation
        kl = stable_kl_divergence(log_probs_base, probs_comp)
        
        # Needle Retrieval (at 1k context)
        # For speed, we only do 1 position if samples=1
        needle = "The secret passkey is 'ALBATROSS-99'."
        question = "What is the secret passkey?"
        answer = "ALBATROSS-99"
        
        # We need a modified run_test that uses our selective compression
        success = self._run_needle_selective(context_len, needle, question, answer, mode, layer_mask, head_mask)
        
        return {
            "kl_divergence": kl,
            "retrieval_success": 1.0 if success else 0.0
        }

    def _run_needle_selective(self, context_len, needle, question, answer, mode, layer_mask, head_mask):
        haystack_ids = self.needle_ev.create_haystack(context_len, needle, 0.5).to(self.device)
        q_text = f"\nQuestion: {question}\nAnswer:"
        q_ids = self.tokenizer(q_text, return_tensors="pt").input_ids.to(self.device)
        
        outputs = self.model(haystack_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        past_kv_recon = self.compress_kv_selective(past_kv, mode, layer_mask=layer_mask, head_mask=head_mask)
        
        outputs = self.model(q_ids, past_key_values=past_kv_recon, use_cache=True)
        curr_past = outputs.past_key_values
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        recon_tokens = []
        for _ in range(15):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id: break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        response = self.tokenizer.decode(recon_tokens, skip_special_tokens=True).strip()
        return verify_retrieval(response, answer)

    @torch.no_grad()
    def analyze_token_importance(self, context_len=1024):
        """
        Task 3: Investigate whether specific token categories require disproportionate protection.
        """
        print("\n>>> Running Token Importance Analysis...")
        # We'll test: Rare tokens, Entities, Delimiters, Repeated tokens
        # We'll do this by measuring reconstruction error (MSE) for these tokens
        
        text = "The quick brown fox jumps over the lazy dog. Einstein lived in Princeton. !!! ??? ### The the the the the."
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        outputs = self.model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values # List of (k, v)
        
        # We'll measure importance by seeing which tokens' KV reconstruction error hurts PPL the most
        # Or more simply: which tokens have the highest 'attention weight' (Task 4)
        
        token_stats = []
        for i, token in enumerate(tokens):
            # Category heuristic
            cat = "Normal"
            if any(c in token for c in "!.?#"): cat = "Delimiter"
            if token[0].isupper(): cat = "Entity/Start"
            if tokens.count(token) > 1: cat = "Repeated"
            
            token_stats.append({"token": token, "pos": i, "category": cat})
            
        return token_stats

    @torch.no_grad()
    def capture_attention_stats(self, text: str):
        """
        Task 4: Add attention-statistics instrumentation.
        """
        print("\n>>> Capturing Attention Statistics...")
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(input_ids, output_attentions=True, return_dict=True)
        attentions = outputs.attentions 
        
        if attentions is None:
            print("[Warning] Model did not return attentions. Semantic Mapping might be limited.")
            return []
            
        # Calculate Entropy and Concentration
        layer_stats = []
        for l, attn in enumerate(attentions):
            # attn: [1, heads, s, s]
            # Average over heads and query positions
            # We care about which TOKENS are attended TO (dim -1)
            mean_attn = attn[0].mean(dim=(0, 1)) # [s]
            
            # Use stable entropy implementation
            # attn: [1, heads, s, s] -> we care about entropy of the last token's attention
            # per head entropy of the last query position
            last_attn = attn[:, :, -1, :] # [1, heads, s]
            head_entropies = stable_attention_entropy(last_attn).squeeze(0).cpu().numpy()
            
            layer_stats.append({
                "layer": l,
                "avg_entropy": float(np.mean(head_entropies)),
                "head_entropies": head_entropies.tolist(),
                "token_centrality": mean_attn.cpu().numpy().tolist()
            })
            
        return layer_stats

    def map_retrieval_circuits(self):
        """
        Task 5: Attempt to identify induction/retrieval heads.
        """
        print("\n>>> Mapping Retrieval Circuits...")
        # Induction heads usually have strong attention to (pos-1) for repeated tokens
        # We'll use a synthetic pattern: " A B C ... A B " -> check if 'B' attends to 'A'
        pattern = " A B C D E A B"
        ids = self.tokenizer(pattern, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(ids, output_attentions=True, return_dict=True)
        attentions = outputs.attentions
        
        if attentions is None:
            return []
        
        # Tokens: [SOS, A, B, C, D, E, A, B]
        # Positions: 0, 1, 2, 3, 4, 5, 6, 7
        # Second 'A' is at 6. It should attend to 'A' at 1 (offset -5) or SOS.
        # Second 'B' is at 7. Induction head: attends to 'A' at 1 (offset -6).
        
        induction_heads = []
        for l, attn in enumerate(attentions):
            for h in range(self.num_heads):
                score = attn[0, h, 7, 1].item() # Attention from pos 7 to pos 1
                if score > 0.3:
                    induction_heads.append((l, h, score))
                    
        return induction_heads

    def generate_report(self, layer_sens, head_sens, attn_stats, induction_heads):
        report_path = Path("results/phase9/Phase9_Semantic_Importance_Report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = "# Phase 9: Semantic Importance Mapping Report\n\n"
        
        content += "## 1. Layer-wise Sensitivity\n"
        content += "| Layer | KL Divergence (↓) | Retrieval Success |\n"
        content += "| :--- | :---: | :---: |\n"
        for r in layer_sens:
            content += f"| {r['layer']} | {r['kl_divergence']:.6f} | {r['retrieval_success']:.0%} |\n"
            
        content += "\n## 2. Top 10 Retrieval-Critical Heads\n"
        content += "| Layer | Head | KL Divergence | Retrieval Impact |\n"
        content += "| :--- | :--- | :---: | :---: |\n"
        # Sort head_sens by kl_divergence descending (more sensitive)
        sorted_heads = sorted(head_sens, key=lambda x: x["kl_divergence"], reverse=True)
        for h in sorted_heads[:10]:
            content += f"| {h['layer']} | {h['head']} | {h['kl_divergence']:.6f} | {'Critical' if h['retrieval_success'] < 1 else 'Stable'} |\n"
            
        content += "\n## 3. Retrieval Circuit Mapping\n"
        content += f"Found {len(induction_heads)} potential induction/retrieval heads.\n"
        for l, h, s in induction_heads[:10]:
            content += f"- Layer {l}, Head {h} (Score: {s:.4f})\n"
            
        content += "\n## 4. Attention Stats Summary\n"
        avg_ent = np.mean([s["avg_entropy"] for s in attn_stats])
        content += f"Average Attention Entropy: {avg_ent:.4f}\n"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n[OK] Phase 9 report generated: {report_path}")

def main():
    analyzer = SemanticImportanceAnalyzer()
    
    # Task 1 & 2
    layer_sens, head_sens = analyzer.evaluate_sensitivity()
    
    # Task 4
    attn_stats = analyzer.capture_attention_stats("The secret password is hidden in the shadows.")
    
    # Task 5
    induction_heads = analyzer.map_retrieval_circuits()
    
    # Generate Report
    analyzer.generate_report(layer_sens, head_sens, attn_stats, induction_heads)
    
    # Save raw data
    with open("results/phase9/raw_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "layer_sensitivity": layer_sens,
            "head_sensitivity": head_sens,
            "attention_stats": attn_stats,
            "induction_heads": induction_heads
        }, f, indent=2)

if __name__ == "__main__":
    main()
