"""
evaluation/generation_eval.py
Phase 8: Generation stability and token divergence evaluation.
Compares compressed generation against FP16 baseline.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.perplexity_eval import Phase8PerplexityEvaluator

class GenerationEvaluator:
    def __init__(self, evaluator: Phase8PerplexityEvaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.tokenizer
        self.device = evaluator.device

    @torch.no_grad()
    def generate_compare(self, prompt: str, mode: str, max_new_tokens: int = 50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Baseline Generation (FP16)
        baseline_out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        baseline_tokens = baseline_out[0, input_ids.size(1):]
        
        # 2. Compressed Generation
        # Split prompt into [prefix] and [last_token]
        prefix_ids = input_ids[:, :-1]
        last_token_id = input_ids[:, -1:]
        
        # Prefill prefix
        outputs = self.model(prefix_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        # Compress
        past_kv_recon, stats = self.ev.compress_kv(past_kv, mode)
        
        # Process last token to get first answer token
        outputs = self.model(last_token_id, past_key_values=past_kv_recon, use_cache=True)
        curr_past = outputs.past_key_values
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        recon_tokens = []
        for _ in range(max_new_tokens):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        # 3. Compare with Behavioral Metrics (KL & Top-K)
        recon_tokens_t = torch.tensor(recon_tokens, device=baseline_tokens.device)
        min_len = min(len(baseline_tokens), len(recon_tokens_t))
        
        matches = (baseline_tokens[:min_len] == recon_tokens_t[:min_len]).float()
        token_overlap = matches.mean().item() if min_len > 0 else 0.0
        
        first_div = -1
        for i in range(min_len):
            if baseline_tokens[i] != recon_tokens_t[i]:
                first_div = i
                break
                
        # Logit Audit for the first token of generation
        with torch.no_grad():
            # Get baseline logits from the original past_kv
            outputs_base = self.model(last_token_id, past_key_values=past_kv, use_cache=True)
            logits_base = F.log_softmax(outputs_base.logits[:, -1, :], dim=-1)
            
            # Get recon logits from the compressed past_kv_recon
            outputs_recon = self.model(last_token_id, past_key_values=past_kv_recon, use_cache=True)
            logits_recon = F.softmax(outputs_recon.logits[:, -1, :], dim=-1)
            
            # Use F.kl_div with log_target=False (default)
            # F.kl_div(input=log_probs, target=probs)
            kl_div = torch.nn.functional.kl_div(logits_base, logits_recon, reduction="batchmean").item()
            
            # Top-k agreement
            top10_base = torch.topk(logits_base, 10).indices[0]
            top10_recon = torch.topk(logits_recon, 10).indices[0]
            topk_agreement = len(set(top10_base.tolist()) & set(top10_recon.tolist())) / 10.0

        return {
            "mode": mode,
            "first_divergence": first_div,
            "token_overlap": token_overlap,
            "kl_divergence": kl_div,
            "topk_agreement": topk_agreement,
            "baseline_text": self.tokenizer.decode(baseline_tokens),
            "recon_text": self.tokenizer.decode(recon_tokens_t),
            "compression_ratio": stats["ratio"]
        }

def run_gen_benchmark(model_id="Qwen/Qwen2-0.5B", n_prompts=3):
    ev = Phase8PerplexityEvaluator(model_id=model_id)
    gen_ev = GenerationEvaluator(ev)
    
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate the Fibonacci sequence.",
        "Once upon a time, in a galaxy far, far away,"
    ]
    
    modes = ["FP16", "INT8-DKV", "Layer-Shared Rank16", "Hybrid-S1%", "Hybrid-S5%"]
    
    results = []
    for prompt in prompts:
        for mode in modes:
            res = gen_ev.generate_compare(prompt, mode)
            results.append(res)
            print(f"Prompt: {prompt[:20]}... | Mode: {mode:20} | Overlap: {res['token_overlap']:6.2%} | Div: {res['first_divergence']}")
            
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--output", type=str, default="results/phase8/generation.json")
    args = parser.parse_args()
    
    res = run_gen_benchmark(model_id=args.model)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
