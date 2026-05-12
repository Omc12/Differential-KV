"""
evaluation/retrieval_eval.py
Phase 8: Multi-document retrieval evaluation.
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.perplexity_eval import Phase8PerplexityEvaluator

class RetrievalEvaluator:
    def __init__(self, evaluator: Phase8PerplexityEvaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.tokenizer
        self.device = evaluator.device

    def get_data(self):
        # Synthetic retrieval task: find the capital of a random country in a list of facts
        facts = [
            "The capital of France is Paris.",
            "The capital of Germany is Berlin.",
            "The capital of Italy is Rome.",
            "The capital of Spain is Madrid.",
            "The capital of Japan is Tokyo.",
            "The capital of China is Beijing.",
            "The capital of Brazil is Brasilia.",
            "The capital of Canada is Ottawa.",
            "The capital of Australia is Canberra.",
            "The capital of India is New Delhi."
        ]
        import random
        random.shuffle(facts)
        context = "\n".join(facts)
        target_fact = random.choice(facts)
        target_country = target_fact.split("of ")[1].split(" is")[0]
        target_city = target_fact.split("is ")[1].replace(".", "")
        
        question = f"What is the capital of {target_country}?"
        return context, question, target_city

    @torch.no_grad()
    def run_test(self, context: str, question: str, answer: str, mode: str):
        context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        q_text = f"\n\nQuestion: {question}\nAnswer:"
        q_ids = self.tokenizer(q_text, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Prefill
        outputs = self.model(context_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        # 2. Compress
        past_kv_recon, _ = self.ev.compress_kv(past_kv, mode)
        
        # 3. Process Question
        outputs = self.model(q_ids, past_key_values=past_kv_recon, use_cache=True)
        curr_past = outputs.past_key_values
        
        # 4. Generate
        recon_tokens = []
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(10):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        response = self.tokenizer.decode(recon_tokens, skip_special_tokens=True)
        success = answer.lower() in response.lower()
        return success, response

def run_retrieval_benchmark(model_id="Qwen/Qwen2-0.5B", n_trials=5):
    ev = Phase8PerplexityEvaluator(model_id=model_id)
    ret_ev = RetrievalEvaluator(ev)
    
    modes = ["FP16", "INT8-DiffKV", "Layer-Shared Rank16", "Hybrid-S1%", "Hybrid-S5%"]
    
    results = {}
    for mode in modes:
        correct = 0
        for _ in range(n_trials):
            context, question, answer = ret_ev.get_data()
            success, resp = ret_ev.run_test(context, question, answer, mode)
            if success:
                correct += 1
        acc = correct / n_trials
        results[mode] = acc
        print(f"Mode: {mode:20} | Retrieval Accuracy: {acc:6.2%}")
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--output", type=str, default="results/phase8/retrieval.json")
    args = parser.parse_args()
    
    res = run_retrieval_benchmark(model_id=args.model)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
