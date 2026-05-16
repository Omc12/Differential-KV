"""
evaluation/needle_haystack.py
Phase 8: Needle-in-a-Haystack evaluation for Differential KV.
"""

import os
import sys
import json
import time
import torch
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.perplexity_eval import Phase8PerplexityEvaluator

class NeedleHaystackEvaluator:
    def __init__(self, evaluator: Phase8PerplexityEvaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.tokenizer
        self.device = evaluator.device

    def create_haystack(self, context_len: int, needle: str, needle_pos: float):
        filler = "The grass is green. The sky is blue. The sun is bright. "
        filler_ids = self.tokenizer(filler, return_tensors="pt").input_ids[0]
        num_filler = context_len // len(filler_ids)
        full_haystack_ids = filler_ids.repeat(num_filler + 1)[:context_len]
        needle_ids = self.tokenizer("\n" + needle + "\n", return_tensors="pt").input_ids[0]
        pos = int(context_len * needle_pos)
        pos = max(0, min(pos, context_len - len(needle_ids)))
        full_haystack_ids[pos:pos+len(needle_ids)] = needle_ids
        return full_haystack_ids.unsqueeze(0)

    @torch.no_grad()
    def run_test(self, context_len: int, needle: str, question: str, answer: str, mode: str, needle_pos: float = 0.5):
        haystack_ids = self.create_haystack(context_len, needle, needle_pos).to(self.device)
        q_text = f"\nQuestion: {question}\nAnswer:"
        q_ids = self.tokenizer(q_text, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Prefill Haystack
        outputs = self.model(haystack_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        # 2. Compress Haystack KV
        past_kv_recon, _ = self.ev.compress_kv(past_kv, mode)
        
        # 3. Process Question
        outputs = self.model(q_ids, past_key_values=past_kv_recon, use_cache=True)
        curr_past = outputs.past_key_values
        
        # 4. Generate Answer (manual loop to avoid 'generate' issues)
        recon_tokens = []
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        for _ in range(20):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        response = self.tokenizer.decode(recon_tokens, skip_special_tokens=True).strip()
        
        # STRICT SUCCESS: Exact match of the passkey ALBATROSS-99
        # Allow only very minor punctuation/whitespace differences
        clean_resp = response.replace("'", "").replace("\"", "").replace(".", "").strip()
        success = answer in clean_resp and len(clean_resp) < 100 # Ensure it's not a hallucinated essay containing the word
        
        # Double check: if it's generic explanation, it should fail
        if "password" in response.lower() and answer not in response:
            success = False
            
        return success, response

def run_needle_benchmark(model_id="Qwen/Qwen2-0.5B", context_lens=[1024, 2048, 4096]):
    ev = Phase8PerplexityEvaluator(model_id=model_id)
    needle_ev = NeedleHaystackEvaluator(ev)
    
    needle = "The secret passkey is 'ALBATROSS-99'."
    question = "What is the secret passkey?"
    answer = "ALBATROSS-99"
    
    modes = ["FP16", "INT8-DiffKV", "Layer-Shared Rank16", "Hybrid-S1%", "Hybrid-S5%"]
    
    results = []
    for cl in context_lens:
        print(f"\n--- Context Length: {cl} ---")
        for mode in modes:
            for pos in [0.1, 0.5, 0.9]:
                success, response = needle_ev.run_test(cl, needle, question, answer, mode, needle_pos=pos)
                results.append({
                    "context_len": cl,
                    "mode": mode,
                    "pos": pos,
                    "success": success,
                    "response": response
                })
                print(f"CL: {cl:5} | Mode: {mode:20} | Pos: {pos:3.1f} | Success: {success} | Resp: {response.strip()}")
                
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--output", type=str, default="results/phase8/needle.json")
    args = parser.parse_args()
    
    res = run_needle_benchmark(model_id=args.model)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
