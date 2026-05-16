import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.perplexity_eval import Phase8PerplexityEvaluator
from evaluation.needle_haystack import NeedleHaystackEvaluator
from evaluation.metrics_utils import verify_retrieval

class RobustRetrievalBenchmark:
    def __init__(self, evaluator: Phase8PerplexityEvaluator):
        self.ev = evaluator
        self.model = evaluator.model
        self.tokenizer = evaluator.tokenizer
        self.device = evaluator.device
        self.needle_ev = NeedleHaystackEvaluator(evaluator)

    def create_multi_needle_haystack(self, context_len: int, needles: List[str], positions: List[float]):
        filler = "The grass is green. The sky is blue. The sun is bright. "
        filler_ids = self.tokenizer(filler, return_tensors="pt").input_ids[0]
        num_filler = context_len // len(filler_ids)
        full_haystack_ids = filler_ids.repeat(num_filler + 1)[:context_len]
        
        for needle, pos_ratio in zip(needles, positions):
            needle_ids = self.tokenizer("\n" + needle + "\n", return_tensors="pt").input_ids[0]
            pos = int(context_len * pos_ratio)
            pos = max(0, min(pos, context_len - len(needle_ids)))
            full_haystack_ids[pos:pos+len(needle_ids)] = needle_ids
            
        return full_haystack_ids.unsqueeze(0)

    @torch.no_grad()
    def run_multi_needle_test(self, context_len: int, needles: List[str], questions: List[str], answers: List[str], mode: str):
        """
        Tests if the model can retrieve multiple distinct needles from the same context.
        """
        positions = [0.1, 0.4, 0.7, 0.9][:len(needles)]
        haystack_ids = self.create_multi_needle_haystack(context_len, needles, positions).to(self.device)
        
        # Prefill & Compress
        outputs = self.model(haystack_ids, use_cache=True)
        past_kv_recon, _ = self.ev.compress_kv(outputs.past_key_values, mode)
        
        results = []
        for q, a in zip(questions, answers):
            q_ids = self.tokenizer(f"\nQuestion: {q}\nAnswer:", return_tensors="pt").input_ids.to(self.device)
            
            outputs = self.model(q_ids, past_key_values=past_kv_recon, use_cache=True)
            curr_past = outputs.past_key_values
            
            recon_tokens = []
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            for _ in range(20):
                recon_tokens.append(next_tok.item())
                if next_tok.item() == self.tokenizer.eos_token_id: break
                outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
                next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_past = outputs.past_key_values
                
            response = self.tokenizer.decode(recon_tokens, skip_special_tokens=True).strip()
            success = verify_retrieval(response, a)
            results.append({"question": q, "success": success, "response": response})
            
        return results

    @torch.no_grad()
    def run_distractor_test(self, context_len: int, needle: str, question: str, answer: str, distractor: str, mode: str):
        """
        Tests retrieval when a 'distractor' (similar but wrong info) is present.
        """
        # Place needle at 0.3, distractor at 0.7
        haystack_ids = self.create_multi_needle_haystack(context_len, [needle, distractor], [0.3, 0.7]).to(self.device)
        
        outputs = self.model(haystack_ids, use_cache=True)
        past_kv_recon, _ = self.ev.compress_kv(outputs.past_key_values, mode)
        
        q_ids = self.tokenizer(f"\nQuestion: {question}\nAnswer:", return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(q_ids, past_key_values=past_kv_recon, use_cache=True)
        
        # Generation loop
        curr_past = outputs.past_key_values
        recon_tokens = []
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(20):
            recon_tokens.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id: break
            outputs = self.model(next_tok, past_key_values=curr_past, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_past = outputs.past_key_values
            
        response = self.tokenizer.decode(recon_tokens, skip_special_tokens=True).strip()
        success = verify_retrieval(response, answer)
        return success, response

    @torch.no_grad()
    def run_induction_test(self, pattern_len: int, mode: str):
        """
        Synthetic induction: ' A B C D ... A ' -> should predict 'B'.
        """
        # Create a random sequence of tokens
        tokens = list(range(100, 200))
        np.random.shuffle(tokens)
        seq = tokens[:pattern_len]
        
        # Pattern: [seq] + [seq[0]]
        full_seq = seq + [seq[0]]
        input_ids = torch.tensor([full_seq]).to(self.device)
        
        # Process first part
        outputs = self.model(input_ids[:, :-1], use_cache=True)
        past_kv_recon, _ = self.ev.compress_kv(outputs.past_key_values, mode)
        
        # Predict last token
        outputs = self.model(input_ids[:, -1:], past_key_values=past_kv_recon, use_cache=True)
        next_tok = outputs.logits[:, -1, :].argmax(dim=-1).item()
        
        return next_tok == seq[1]

def main():
    ev = Phase8PerplexityEvaluator(model_id="Qwen/Qwen2-0.5B")
    benchmark = RobustRetrievalBenchmark(ev)
    
    needles = [
        "The secret passkey is 'ALBATROSS-99'.",
        "The manager's name is 'BARNABY-WHEELS'.",
        "The office is located in 'SECTOR-7G'."
    ]
    questions = [
        "What is the secret passkey?",
        "What is the manager's name?",
        "Where is the office located?"
    ]
    answers = ["ALBATROSS-99", "BARNABY-WHEELS", "SECTOR-7G"]
    
    modes = ["FP16", "Layer-Shared Rank8", "Hybrid-S5%"]
    
    results = {}
    for mode in modes:
        print(f"\n>>> Running Robust Retrieval (Mode: {mode})...")
        
        # 1. Multi-needle
        multi_res = benchmark.run_multi_needle_test(2048, needles, questions, answers, mode)
        
        # 2. Distractor
        dist_success, dist_resp = benchmark.run_distractor_test(
            2048, 
            needles[0], 
            questions[0], 
            answers[0], 
            "The fake passkey is 'EAGLE-11'.", 
            mode
        )
        
        # 3. Induction
        ind_success = benchmark.run_induction_test(128, mode)
        
        results[mode] = {
            "multi_needle": multi_res,
            "distractor": {"success": dist_success, "response": dist_resp},
            "induction": ind_success
        }
        
    # Save results
    os.makedirs("results/phase9", exist_ok=True)
    with open("results/phase9/robust_retrieval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[OK] Robust Retrieval Benchmark complete.")

if __name__ == "__main__":
    main()
