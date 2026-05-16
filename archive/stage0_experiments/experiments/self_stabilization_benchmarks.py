"""
experiments/self_stabilization_benchmarks.py
Phase 17: Self-Stabilization Benchmarks
Evaluates reasoning survival WITHOUT external repair across multiple tasks:
GSM8K, multi-hop reasoning, and trajectory persistence.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import numpy as np

class SelfStabilizationBenchmarker:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device

    def run_benchmark(self, tasks: List[Dict], noise_level=0.05):
        print(f"\nRunning Self-Stabilization Benchmark (Noise: {noise_level})...")
        
        results = []
        for task in tqdm(tasks):
            prompt = task["prompt"]
            answer = task["answer"]
            
            # Run generation with noise
            def hook_fn(module, input, output):
                return output + torch.randn_like(output) * noise_level
                
            hooks = []
            for layer in self.model.model.layers:
                hooks.append(layer.register_forward_hook(hook_fn))
                
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            for h in hooks: h.remove()
            
            # Check survival (does it contain the correct answer?)
            survival = answer.lower() in generated_text.lower()
            
            results.append({
                "prompt": prompt,
                "generated": generated_text,
                "answer": answer,
                "survival": survival
            })
            
        accuracy = sum([r["survival"] for r in results]) / len(results)
        return accuracy, results

if __name__ == "__main__":
    benchmarker = SelfStabilizationBenchmarker()
    
    # Mock GSM8K/Reasoning tasks
    tasks = [
        {"prompt": "If John has 5 apples and eats 2, how many are left?", "answer": "3"},
        {"prompt": "What is the next prime number after 7?", "answer": "11"},
        {"prompt": "Solve for x: 2x = 10.", "answer": "5"},
        {"prompt": "Who wrote 'Romeo and Juliet'?", "answer": "Shakespeare"},
        {"prompt": "What is the capital of Japan?", "answer": "Tokyo"}
    ]
    
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    bench_results = {}
    
    for noise in noise_levels:
        acc, details = benchmarker.run_benchmark(tasks, noise_level=noise)
        bench_results[f"noise_{noise}"] = acc
        print(f"Noise {noise} Accuracy: {acc:.2f}")
        
    os.makedirs("results/phase17/data", exist_ok=True)
    with open("results/phase17/data/self_stabilization_benchmarks.json", "w") as f:
        json.dump(bench_results, f, indent=4)
