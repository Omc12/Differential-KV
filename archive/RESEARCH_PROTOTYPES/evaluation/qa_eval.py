"""
evaluation/qa_eval.py
Phase 8: Question Answering Fidelity (SQuAD-style).
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.perplexity_eval import Phase8PerplexityEvaluator

class QAEvaluator:
    def __init__(self, evaluator: Phase8PerplexityEvaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.tokenizer
        self.device = evaluator.device

    def get_data(self, n_samples=5):
        dataset = load_dataset("squad", split="validation")
        samples = []
        for i in range(n_samples):
            samples.append({
                "context": dataset[i]["context"],
                "question": dataset[i]["question"],
                "answers": dataset[i]["answers"]["text"]
            })
        return samples

    @torch.no_grad()
    def run_test(self, context: str, question: str, answers: List[str], mode: str):
        context_ids = self.tokenizer(f"Context: {context}\n\n", return_tensors="pt").input_ids.to(self.device)
        q_text = f"Question: {question}\nAnswer:"
        q_ids = self.tokenizer(q_text, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Prefill context
        outputs = self.model(context_ids, use_cache=True)
        past_kv = outputs.past_key_values
        
        # 2. Compress context
        past_kv_recon, _ = self.ev.compress_kv(past_kv, mode)
        
        # 3. Process question
        outputs = self.model(q_ids, past_key_values=past_kv_recon, use_cache=True)
        curr_past = outputs.past_key_values
        
        # 4. Generate answer
        recon_tokens = []
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(20):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        response = self.tokenizer.decode(recon_tokens, skip_special_tokens=True)
        success = any(a.lower() in response.lower() for a in answers)
        return success, response

def run_qa_benchmark(model_id="Qwen/Qwen2-0.5B", n_samples=5):
    ev = Phase8PerplexityEvaluator(model_id=model_id)
    qa_ev = QAEvaluator(ev)
    
    samples = qa_ev.get_data(n_samples=n_samples)
    modes = ["FP16", "INT8-DiffKV", "Layer-Shared Rank16", "Hybrid-S1%", "Hybrid-S5%"]
    
    results = {}
    for mode in modes:
        correct = 0
        for s in samples:
            success, resp = qa_ev.run_test(s["context"], s["question"], s["answers"], mode)
            if success:
                correct += 1
        acc = correct / len(samples)
        results[mode] = acc
        print(f"Mode: {mode:20} | QA Accuracy: {acc:6.2%}")
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--output", type=str, default="results/phase8/qa.json")
    args = parser.parse_args()
    
    res = run_qa_benchmark(model_id=args.model)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
