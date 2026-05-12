"""
evaluation/phase9_drift_and_protection.py
Phase 9: Semantic Drift and Selective Protection Simulation.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.phase9_semantic_mapping import SemanticImportanceAnalyzer
from evaluation.metrics_utils import (
    stable_kl_divergence, 
    stable_cosine_drift, 
    logit_rank_correlation, 
    top_k_overlap
)

class DriftAndProtectionAnalyzer:
    def __init__(self, analyzer: SemanticImportanceAnalyzer):
        self.analyzer = analyzer
        self.model = analyzer.model
        self.tokenizer = analyzer.tokenizer
        self.device = analyzer.device

    @torch.no_grad()
    def analyze_semantic_drift_over_time(self, prompt: str, mode: str, max_tokens: int = 30):
        """
        Task 6 & 7: Identify where semantic drift begins.
        Measures KL, Cosine, Rank correlation at each step.
        """
        print(f"\n>>> Analyzing Semantic Drift over time (Mode: {mode})...")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Baseline Generation (FP16) - save logits and hidden states
        baseline_data = []
        curr_ids = input_ids.clone()
        past_kv = None
        for _ in range(max_tokens):
            outputs = self.model(
                curr_ids if past_kv is None else curr_ids[:, -1:], 
                past_key_values=past_kv, 
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            hidden = outputs.hidden_states[-1][:, -1, :] # Last layer hidden state
            
            baseline_data.append({
                "log_probs": F.log_softmax(logits, dim=-1),
                "logits": logits,
                "hidden": hidden,
                "token": logits.argmax(dim=-1, keepdim=True)
            })
            
            next_tok = logits.argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
            if next_tok.item() == self.tokenizer.eos_token_id: break
            
        # 2. Compressed Generation - force follow baseline tokens to measure logit drift
        drift_stats = []
        
        # Prefill input_ids
        outputs = self.model(input_ids, use_cache=True)
        # Compress the prefilled KV
        past_kv_comp, _ = self.analyzer.ev.compress_kv(outputs.past_key_values, mode)
        
        curr_past = past_kv_comp
        
        for i in range(len(baseline_data)):
            # We want to see how the model behaves at step i, 
            # GIVEN the context of input_ids + baseline_tokens[:i]
            
            # If i > 0, we need to feed the PREVIOUS baseline token
            if i > 0:
                prev_tok = baseline_data[i-1]["token"]
                outputs = self.model(
                    prev_tok, 
                    past_key_values=curr_past, 
                    use_cache=True, 
                    output_hidden_states=True
                )
                curr_past = outputs.past_key_values
            else:
                # For i=0, we are at the end of prefill. 
                # HF models usually don't need a call if we already have the logits, 
                # but we need the hidden state from the COMPRESSED path.
                # Re-run the last token of input_ids to get compressed hidden state?
                # Actually, a cleaner way is to just run from the last token.
                outputs = self.model(input_ids[:, -1:], past_key_values=None, use_cache=True)
                # This is messy. Let's just do it step by step from i=0.
                # Actually, the simplest is:
                # The first 'logits_comp' comes from the last token of prefill.
                # But we compressed the KV *after* prefilling.
                # To see the effect of compression, we must re-evaluate.
                pass
            
            # At this point, curr_past contains KV for input_ids + baseline_tokens[:i]
            # EXCEPT for the very first step where it's just input_ids.
            # We want the logits for the token at position len(input_ids) + i.
            # This is predicted by the token at len(input_ids) + i - 1.
            
            # Let's simplify: 
            # Baseline token i is predicted by (input_ids + baseline_tokens[:i])
            # So we feed baseline_tokens[i-1] (or last of input_ids if i=0)
            # and get logits for i.
            
            # wait, if i=0, we already have KV for input_ids[0:N]. 
            # HF cache usually stores KV for [0:N]. 
            # To get logits for token N, we need to have processed [0:N-1] 
            # and then pass token N-1.
            
            # Let's just re-run the entire input_ids through the compression if i=0?
            # No, that's slow.
            
            # RE-TRYING DRIFT LOGIC:
            # 1. Prefill [0:N-1], compress.
            # 2. Loop i from 0 to max_tokens:
            #    - feed token (N-1 + i), get logits for (N+i)
            #    - compare with baseline logits for (N+i)
            
            # Initial setup for i=0
            if i == 0:
                # We need KV for input_ids[:, :-1]
                out_pre = self.model(input_ids[:, :-1], use_cache=True)
                curr_past, _ = self.analyzer.ev.compress_kv(out_pre.past_key_values, mode)
                target_in = input_ids[:, -1:]
            else:
                target_in = baseline_data[i-1]["token"]
            
            outputs = self.model(target_in, past_key_values=curr_past, use_cache=True, output_hidden_states=True)
            curr_past = outputs.past_key_values
            
            logits_comp = outputs.logits[:, -1, :]
            probs_comp = F.softmax(logits_comp, dim=-1)
            hidden_comp = outputs.hidden_states[-1][:, -1, :]
            
            # Metrics
            kl = stable_kl_divergence(baseline_data[i]["log_probs"], probs_comp)
            cos = stable_cosine_drift(baseline_data[i]["hidden"], hidden_comp)
            corr = logit_rank_correlation(baseline_data[i]["logits"], logits_comp)
            overlap = top_k_overlap(baseline_data[i]["logits"], logits_comp)
            
            drift_stats.append({
                "step": i,
                "kl": kl,
                "cosine": cos,
                "rank_corr": corr,
                "top10_overlap": overlap
            })
            
        return drift_stats

    @torch.no_grad()
    def run_selective_protection_experiment(self, critical_layers: Set[int], mode="Layer-Shared Rank8"):
        """
        Task 7: Test whether semantic-aware preservation beats uniform sparse repair.
        """
        print(f"\n>>> Running Selective Protection Experiment (Protecting layers: {critical_layers})...")
        
        # 1. Uniform Compression (everything compressed)
        uniform_results = self.analyzer._get_quick_metrics(mode, samples=5, context_len=2048)
        
        # 2. Selective Protection (everything compressed EXCEPT critical_layers)
        # In our compress_kv_selective, layer_mask=None means all. 
        # To compress all EXCEPT some, we need to pass the set of ALL layers MINUS critical.
        all_layers = set(range(self.analyzer.num_layers))
        comp_layers = all_layers - critical_layers
        
        selective_results = self.analyzer._get_quick_metrics(mode, layer_mask=comp_layers, samples=5, context_len=2048)
        
        return {
            "uniform": uniform_results,
            "selective": selective_results
        }

def main():
    mapping_data = json.load(open("results/phase9/raw_data.json", "r"))
    layer_sens = mapping_data["layer_sensitivity"]
    
    # Identify critical layers (e.g. top 2 by KL)
    sorted_layers = sorted(layer_sens, key=lambda x: x["kl_divergence"], reverse=True)
    critical_layers = {sorted_layers[0]["layer"], sorted_layers[1]["layer"]}
    
    analyzer = SemanticImportanceAnalyzer()
    drifter = DriftAndProtectionAnalyzer(analyzer)
    
    # Task 6
    prompt = "The future of artificial intelligence lies in its ability to understand"
    drift_kl = drifter.analyze_semantic_drift_over_time(prompt, "Layer-Shared Rank16")
    
    # Task 7
    prot_results = drifter.run_selective_protection_experiment(critical_layers)
    
    # Save results
    with open("results/phase9/drift_and_protection.json", "w") as f:
        json.dump({
            "drift_kl": drift_kl,
            "protection_experiment": prot_results,
            "protected_layers": list(critical_layers)
        }, f, indent=2)
        
    # Append to Report
    report_path = "results/phase9/Phase9_Semantic_Importance_Report.md"
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 5. Semantic Drift Analysis (Stable)\n")
        f.write(f"Prompt: '{prompt}'\n")
        f.write("| Step | KL Div | Cosine | Rank Corr | Top10 Overlap |\n| :--- | :---: | :---: | :---: | :---: |\n")
        for stat in drift_kl:
            f.write(f"| {stat['step']} | {stat['kl']:.6f} | {stat['cosine']:.4f} | {stat['rank_corr']:.4f} | {stat['top10_overlap']:.1%} |\n")
            
        f.write("\n## 6. Selective Protection Simulation (Audit)\n")
        f.write(f"Protected Layers: {critical_layers}\n")
        f.write("| Strategy | KL Divergence | Retrieval Success |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| Uniform Rank8 | {prot_results['uniform']['kl_divergence']:.6f} | {prot_results['uniform']['retrieval_success']:.0%} |\n")
        f.write(f"| Selective Rank8 | {prot_results['selective']['kl_divergence']:.6f} | {prot_results['selective']['retrieval_success']:.0%} |\n")
        
    print("\n[OK] Phase 9 drift and protection analysis complete.")

if __name__ == "__main__":
    main()
