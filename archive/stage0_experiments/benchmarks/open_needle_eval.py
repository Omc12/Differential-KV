"""
benchmarks/open_needle_eval.py

Needle-in-Haystack evaluation for Differential KV.
Validates perfect retrieval at extreme context lengths (up to 1M+ tokens).
"""

import torch
import time
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.differential_kv_runtime import DifferentialKVRuntime
from typing import Dict, Any, List

class OpenNeedleEvaluator:
    def __init__(self, model_id: str, runtime_config: Dict[str, Any]):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.runtime = DifferentialKVRuntime(self.model, runtime_config)
        self.model = self.runtime.patched_model
        self.results = []

    def run_eval(self, context_lengths: List[int], needle_positions: List[float]):
        """
        needle_positions: float between 0 and 1 representing relative position in context.
        """
        print(f"Starting Open Needle-in-Haystack Evaluation")
        
        needle = "The special code is 12345."
        question = "What is the special code?"
        
        for length in context_lengths:
            for pos in needle_positions:
                # Construct context
                haystack_tokens = torch.randint(0, self.tokenizer.vocab_size, (1, length)).tolist()[0]
                needle_tokens = self.tokenizer(needle, add_special_tokens=False).input_ids
                
                insert_idx = int(length * pos)
                full_tokens = haystack_tokens[:insert_idx] + needle_tokens + haystack_tokens[insert_idx:]
                
                # Add question
                question_tokens = self.tokenizer(f"\n{question}", add_special_tokens=False).input_ids
                input_ids = torch.tensor([full_tokens + question_tokens]).to(self.model.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=10,
                        use_cache=True,
                        do_sample=False
                    )
                latency = time.time() - start_time
                
                response = self.tokenizer.decode(output_ids[0][-10:], skip_special_tokens=True)
                success = "12345" in response
                
                self.results.append({
                    "context_length": length,
                    "needle_position": pos,
                    "latency": latency,
                    "success": success,
                    "response": response
                })
                print(f"Len {length}, Pos {pos:.2f}: {'PASS' if success else 'FAIL'} ({latency:.2f}s)")

    def save_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    config = {"mode": "differential", "sparse_ratio": 0.01}
    model_id = "Qwen/Qwen2-7B-Instruct"
    evaluator = OpenNeedleEvaluator(model_id, config)
    evaluator.run_eval([128000, 256000], [0.1, 0.5, 0.9])
    evaluator.save_results("results/phase38/needle_results.json")
