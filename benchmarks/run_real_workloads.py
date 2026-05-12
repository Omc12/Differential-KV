"""
benchmarks/run_real_workloads.py
Phase 13 Task 8: Real Workload Analysis
Tests SAM on coding, multi-hop reasoning, and long planning tasks.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import json
import matplotlib.pyplot as plt

class RealWorkloadAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2-0.5B", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        
    @torch.no_grad()
    def run_workload(self, name, text, question, answer):
        print(f"\n>>> Running Workload: {name}...")
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        context_len = input_ids.shape[1]
        
        # 1. Baseline Retrieval (FP16)
        out_base = self.model(input_ids, use_cache=True)
        kv_base = out_base.past_key_values
        
        # 2. Compressed Retrieval
        # Simulate high compression rank or noise
        noise_std = 0.2
        kv_noisy = []
        if hasattr(kv_base, "to_legacy_cache"): kv_base = kv_base.to_legacy_cache()
        for k, v in kv_base:
            kv_noisy.append((k + torch.randn_like(k) * noise_std, v + torch.randn_like(v) * noise_std))
        
        # 3. SAM Retrieval
        # Anchor every 64 tokens + first 4 tokens
        kv_sam = []
        anchors = [0, 1, 2, 3] + list(range(0, context_len, 64))
        for l in range(len(kv_noisy)):
            k, v = kv_noisy[l][0].clone(), kv_noisy[l][1].clone()
            for p in anchors:
                if p < context_len:
                    k[:, :, p, :] = kv_base[l][0][:, :, p, :]
                    v[:, :, p, :] = kv_base[l][1][:, :, p, :]
            kv_sam.append((k, v))
            
        # Test generation
        q_ids = self.tokenizer(f"\nQuestion: {question}\nAnswer:", return_tensors="pt").input_ids.to(self.device)
        
        resp_noisy = self._generate(q_ids, tuple(kv_noisy))
        resp_sam = self._generate(q_ids, tuple(kv_sam))
        
        print(f"  Noisy Response: {resp_noisy}")
        print(f"  SAM Response:   {resp_sam}")
        
        return {
            "name": name,
            "noisy_correct": answer.lower() in resp_noisy.lower(),
            "sam_correct": answer.lower() in resp_sam.lower()
        }

    def _generate(self, q_ids, kv):
        cache = DynamicCache.from_legacy_cache(kv)
        curr_ids = q_ids
        generated = []
        for _ in range(15):
            outputs = self.model(curr_ids, past_key_values=cache, use_cache=True)
            next_tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            curr_ids = next_tok
            cache = outputs.past_key_values
        return self.tokenizer.decode(generated, skip_special_tokens=True)

if __name__ == "__main__":
    analyzer = RealWorkloadAnalyzer()
    
    workloads = [
        {
            "name": "Coding (Python)",
            "text": "def calculate_orbit(mass, velocity, radius):\n    # G is the gravitational constant\n    G = 6.67430e-11\n    return (G * mass * velocity) / radius**2",
            "question": "What is the value of G used in the function?",
            "answer": "6.67430e-11"
        },
        {
            "name": "Multi-hop Reasoning",
            "text": "Alice lives in Paris. Bob lives in London. Charlie lives in the same city as Alice. David lives in the same city as Bob.",
            "question": "In which city does Charlie live?",
            "answer": "Paris"
        },
        {
            "name": "Planning",
            "text": "Step 1: Buy ingredients. Step 2: Preheat oven. Step 3: Mix batter. Step 4: Bake for 30 minutes. Step 5: Let it cool.",
            "question": "What is Step 3?",
            "answer": "Mix batter"
        }
    ]
    
    results = []
    for w in workloads:
        results.append(analyzer.run_workload(w["name"], w["text"], w["question"], w["answer"]))
        
    os.makedirs("results/phase13", exist_ok=True)
    with open("results/phase13/real_workload_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Simple report
    print("\n=== Real Workload Summary ===")
    for r in results:
        print(f"{r['name']}: Noisy={'OK' if r['noisy_correct'] else 'FAIL'}, SAM={'OK' if r['sam_correct'] else 'FAIL'}")
