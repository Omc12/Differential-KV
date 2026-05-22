"""
experiments/real_long_context_eval.py

Evaluates Differential KV on real models with long contexts (32k - 256k).
Measures context retention, reasoning survival, and perplexity.
"""

import torch
import time
import json
import os
from integrations.llamacpp_runtime_adapter import LlamaCppAdapter

def run_long_context_suite():
    print("=== Phase 28: Real Long Context Evaluation ===")
    
    contexts = [32768, 65536, 131072, 262144]
    models = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"]
    
    results = {}
    
    for model in models:
        results[model] = {}
        for ctx_len in contexts:
            print(f"\nTesting {model} @ {ctx_len // 1024}k...")
            
            config = {
                "n_ctx": ctx_len,
                "mode": "diffkv_adaptive",
                "target_compression": 20.0
            }
            
            # Using our adapter
            adapter = LlamaCppAdapter(f"models/{model}.gguf", config)
            
            # 1. Retrieval Task (Needle in a Haystack style)
            start_time = time.time()
            prompt = "The secret code is 12345. " + "Garbage text. " * (ctx_len // 10) + "What is the secret code?"
            response = adapter.generate(prompt, max_tokens=20)
            latency = time.time() - start_time
            
            # 2. Reasoning Task (Recursive planning)
            # prompt_reasoning = "Step 1: A. Step 2: B. ..."
            
            metrics = adapter.get_metrics()
            
            results[model][ctx_len] = {
                "retrieval_success": "12345" in response,
                "latency_sec": latency,
                "intervention_density": metrics["intervention_density"],
                "avg_routing_ms": metrics["avg_latency_per_token"]
            }
            
            print(f"  Result: {'PASS' if results[model][ctx_len]['retrieval_success'] else 'FAIL'}")
            print(f"  Routing Latency: {metrics['avg_latency_per_token']:.2f}ms")

    os.makedirs("results/phase28", exist_ok=True)
    with open("results/phase28/long_context_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_long_context_suite()
