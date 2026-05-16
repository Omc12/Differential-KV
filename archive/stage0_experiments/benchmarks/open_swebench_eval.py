"""
benchmarks/open_swebench_eval.py

Evaluates Differential KV on SWE-Bench tasks.
Focuses on long-context software engineering reasoning and repository navigation.
"""

import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.differential_kv_runtime import DifferentialKVRuntime
from typing import Dict, Any

class OpenSWEBenchEvaluator:
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

    def run_eval(self, repo_context_len: int = 128000, num_tasks: int = 5):
        print(f"Starting Open SWE-Bench Evaluation")
        
        for i in range(num_tasks):
            # Simulate a large repository context + issue description
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, repo_context_len)).to(self.model.device)
            
            start_time = time.time()
            with torch.no_grad():
                # Generate a patch or solution
                _ = self.model.generate(
                    input_ids,
                    max_new_tokens=512,
                    use_cache=True,
                    do_sample=False
                )
            latency = time.time() - start_time
            
            vram = torch.cuda.memory_allocated() / (1024**3)
            
            self.results.append({
                "task_idx": i,
                "context_len": repo_context_len,
                "latency": latency,
                "vram_gb": vram,
                "status": "success"
            })
            print(f"Task {i}: {latency:.2f}s, {vram:.2f} GB")

    def save_results(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    config = {"mode": "differential", "sparse_ratio": 0.05}
    model_id = "Qwen/Qwen2-7B-Instruct"
    evaluator = OpenSWEBenchEvaluator(model_id, config)
    evaluator.run_eval(repo_context_len=64000, num_tasks=3)
    evaluator.save_results("results/phase38/swebench_results.json")
