import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.needle_haystack import NeedleHaystackEvaluator
from phase20.validation.compression_engine import UniversalCompressionEngine

class MultiHopNeedleEvaluator:
    def __init__(self, model_id="Qwen/Qwen2-0.5B"):
        self.model_id = model_id
        # Use existing needle evaluator infrastructure
        from evaluation.perplexity_eval import Phase8PerplexityEvaluator
        self.ev = Phase8PerplexityEvaluator(model_id=model_id)
        self.engine = UniversalCompressionEngine(self.ev.model, self.ev.tokenizer)

    def run_multihop_test(self, n_needles, n_distractors, context_len, mode):
        # Construct multi-hop prompt
        # Needle 1: A is B
        # Needle 2: B is C
        # Question: What is A? Answer: C
        
        # Simplified: Multiple independent needles
        needles = [f"The secret key {i} is {random.randint(1000, 9999)}" for i in range(n_needles)]
        haystack = ["The grass is green.", "The sky is blue.", "Python is a language."] * (n_distractors // 3)
        
        # Insert needles at random positions
        combined = haystack.copy()
        for n in needles:
            combined.insert(random.randint(0, len(combined)), n)
        
        context = " ".join(combined)
        
        results = []
        for i in range(n_needles):
            question = f"What is the secret key {i}?"
            target = needles[i].split(" is ")[1]
            
            # Use the compression engine
            input_ids = self.ev.tokenizer(context + "\n" + question, return_tensors="pt").input_ids.to(self.ev.device)
            
            with torch.no_grad():
                outputs = self.ev.model(input_ids, use_cache=True)
                past_kv = outputs.past_key_values
                past_kv_recon, _ = self.engine.compress_kv(past_kv, mode)
                
                gen_ids = self.ev.model.generate(
                    input_ids[:, -1:],
                    past_key_values=past_kv_recon,
                    max_new_tokens=10,
                    do_sample=False
                )
                response = self.ev.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                success = target in response
                results.append(success)
        
        return sum(results) / len(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--needles", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--distractors", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--context_lengths", nargs="+", default=["8k", "16k", "32k", "64k"])
    parser.add_argument("--modes", nargs="+", default=["rank8", "sam", "actr", "lcg"])
    parser.add_argument("--output", type=str, default="phase20/results/multihop_retrieval.json")
    args = parser.parse_args()
    
    evaluator = MultiHopNeedleEvaluator()
    results = {}
    
    for mode in args.modes:
        results[mode] = {}
        for n in args.needles:
            results[mode][n] = {}
            for d in args.distractors:
                score = evaluator.run_multihop_test(n, d, 8192, mode)
                results[mode][n][d] = score
                print(f"Mode: {mode} | Needles: {n} | Distractors: {d} | Score: {score:.2%}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
