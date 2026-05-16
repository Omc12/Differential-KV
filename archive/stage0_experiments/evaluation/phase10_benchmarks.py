"""
evaluation/phase10_benchmarks.py
Phase 10: Expanded Semantic Stability Benchmarks.
Multi-needle, long-form continuity, and exact retrieval survival.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.metrics_utils import verify_retrieval, stable_kl_divergence, top_k_overlap

class Phase10BenchmarkSuite:
    def __init__(self, evaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.tokenizer
        self.device = evaluator.device
        
    def multi_needle_retrieval(self, context_len: int, num_needles: int = 3) -> Dict[str, Any]:
        """
        Tests retrieval of multiple distinct facts hidden in a large haystack.
        """
        print(f"\n>>> Running Multi-Needle Retrieval (Ctx: {context_len}, Needles: {num_needles})...")
        
        needles = [
            ("The secret code for project Alpha is 'XJ-42'.", "What is the secret code for project Alpha?", "XJ-42"),
            ("The capital of planet Mars is 'Valles City'.", "What is the capital of Mars?", "Valles City"),
            ("The CEO of Antigravity is 'Dr. Neural'.", "Who is the CEO of Antigravity?", "Dr. Neural"),
            ("The favorite food of penguins is 'glacier sushi'.", "What is the favorite food of penguins?", "glacier sushi")
        ]
        
        selected_needles = needles[:num_needles]
        
        # Build haystack with multiple needles at different depths
        depths = np.linspace(0.1, 0.9, num_needles)
        
        # We'll use a base haystack of random tokens
        # For simplicity, we reuse the NeedleHaystackEvaluator's logic but for multiple insertions
        full_text = "Background context: " + "Long sentences about nothing specific. " * (context_len // 10)
        
        # Insert needles
        text_list = full_text.split()
        for i, (n_text, _, _) in enumerate(selected_needles):
            pos = int(len(text_list) * depths[i])
            text_list.insert(pos, n_text)
            
        haystack_ids = self.tokenizer(" ".join(text_list), return_tensors="pt", truncation=True, max_length=context_len).input_ids.to(self.device)
        
        # Baseline Pass
        outputs = self.model(haystack_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        results = []
        for i, (n_text, q_text, a_text) in enumerate(selected_needles):
            q_ids = self.tokenizer(f"\nQuestion: {q_text}\nAnswer:", return_tensors="pt").input_ids.to(self.device)
            
            # We'll test with the provided KV (could be compressed externally)
            # For now, this is a baseline helper
            resp = self._generate_response(q_ids, past_kv)
            success = verify_retrieval(resp, a_text)
            results.append({"needle_idx": i, "success": success, "response": resp})
            
        return {
            "num_needles": num_needles,
            "success_rate": sum(r["success"] for r in results) / num_needles,
            "details": results
        }

    def long_form_continuity(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Tests autoregressive generation stability over many tokens.
        Tracks KL drift and semantic cliff onset.
        """
        print(f"\n>>> Running Long-form Continuity Test (Max tokens: {max_tokens})...")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        tokens = []
        curr_ids = input_ids.clone()
        past_kv = None
        
        # We need a reference generation (FP16)
        # But we'll run it in parallel or use it as a probe
        # For Phase 10, we want to measure how long the COMPRESSED model stays coherent
        
        return {"prompt": prompt, "max_tokens": max_tokens} # Placeholder for actual loop in main script

    def _generate_response(self, q_ids, past_kv, max_new_tokens=15):
        curr_past = past_kv
        next_tok = self.model(q_ids, past_key_values=curr_past, use_cache=True).logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        recon_tokens = []
        curr_past = self.model(q_ids, past_key_values=curr_past, use_cache=True).past_key_values
        
        for _ in range(max_new_tokens):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id: break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        return self.tokenizer.decode(recon_tokens, skip_special_tokens=True).strip()
