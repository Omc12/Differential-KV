import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.needle_haystack import NeedleHaystackEvaluator
from compression.universal_engine import UniversalCompressionEngine

class MultiHopNeedleEvaluator:
    def __init__(self, model_id="Qwen/Qwen2-0.5B"):
        from evaluation.perplexity_eval import Phase8PerplexityEvaluator
        self.ev = Phase8PerplexityEvaluator(model_id=model_id)
        self.engine = UniversalCompressionEngine(self.ev.model, self.ev.tokenizer)

    def run_multihop_test(self, n_needles, n_distractors, context_len, mode):
        needles = [f"The secret key {i} is {random.randint(1000, 9999)}" for i in range(n_needles)]
        haystack = ["The grass is green.", "The sky is blue."] * (n_distractors // 2)
        combined = haystack.copy()
        for n in needles: combined.insert(random.randint(0, len(combined)), n)
        context = " ".join(combined)
        results = []
        for i in range(n_needles):
            question = f"What is the secret key {i}?"
            target = needles[i].split(" is ")[1]
            input_ids = self.ev.tokenizer(context + "\n" + question, return_tensors="pt").input_ids.to(self.ev.device)
            with torch.no_grad():
                prefix_ids = input_ids[:, :-1]
                outputs = self.ev.model(prefix_ids, use_cache=True)
                past_kv_recon, _ = self.engine.compress_kv(outputs.past_key_values, mode)
                last_token_id = input_ids[:, -1:]
                outputs = self.ev.model(last_token_id, past_key_values=past_kv_recon, use_cache=True)
                next_tok, curr_kv = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True), outputs.past_key_values
                gen = []
                for _ in range(10):
                    gen.append(next_tok.item())
                    outputs = self.ev.model(next_tok, past_key_values=curr_kv, use_cache=True)
                    next_tok, curr_kv = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True), outputs.past_key_values
                success = target in self.ev.tokenizer.decode(gen)
                results.append(success)
        return sum(results) / len(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--needles", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--distractors", nargs="+", type=int, default=[64])
    parser.add_argument("--modes", nargs="+", default=["rank8", "lcg"])
    parser.add_argument("--output", type=str, default="results/phase20/multihop_retrieval.json")
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
    with open(args.output, "w") as f: json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
