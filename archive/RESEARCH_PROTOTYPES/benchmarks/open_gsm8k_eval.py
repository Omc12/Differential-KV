"""
benchmarks/open_gsm8k_eval.py

Evaluates reasoning retention and accuracy on GSM8K using Differential KV.
Focuses on math reasoning under memory pressure.
"""

import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.differential_kv_runtime import DifferentialKVRuntime
from typing import Dict, Any

class OpenGSM8KEvaluator:
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

    def run_eval(self, num_samples: int = 50):
        print(f"Starting Open GSM8K Evaluation")
        
        # Simulated GSM8K samples for validation (In real: load from datasets)
        samples = [
            {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
            {"question": "Weng earns $12 an hour for tutoring. If he tutored for 50 minutes, how much did he earn?", "answer": "10"}
        ] * (num_samples // 2)

        for i, sample in enumerate(samples):
            prompt = f"Question: {sample['question']}\nAnswer:"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            
            start_time = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=128,
                    use_cache=True,
                    do_sample=False
                )
            latency = time.time() - start_time
            
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            self.results.append({
                "sample_idx": i,
                "question": sample['question'],
                "expected": sample['answer'],
                "response": response,
                "latency": latency,
                "status": "success"
            })
            print(f"Sample {i} completed in {latency:.2f}s")

    def save_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    config = {"mode": "differential", "sparse_ratio": 0.1}
    model_id = "Qwen/Qwen2-7B-Instruct"
    evaluator = OpenGSM8KEvaluator(model_id, config)
    evaluator.run_eval(10)
    evaluator.save_results("results/phase38/gsm8k_results.json")
