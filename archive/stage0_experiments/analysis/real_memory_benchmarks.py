import os
import sys
import json
import torch
import time
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.universal_engine import UniversalCompressionEngine

def measure_memory_metrics(models, modes, output_path):
    results = {}
    for model_id in models:
        print(f"\n>>> Profiling Model: {model_id}")
        model_map = {"qwen2-1.5b": "Qwen/Qwen2-1.5B", "llama3-8b": "meta-llama/Meta-Llama-3-8B", "mistral-7b": "mistralai/Mistral-7B-v0.1"}
        hf_id = model_map.get(model_id, model_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            engine = UniversalCompressionEngine(model, tokenizer)
        except Exception as e:
            print(f"Error loading {model_id}: {e}"); continue
        results[model_id] = {}
        ctx_len = 8192; input_ids = torch.randint(0, 1000, (1, ctx_len), device=model.device)
        for mode in modes:
            print(f"  Measuring Mode: {mode}"); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
                past_kv = outputs.past_key_values
                comp_start = time.time(); past_kv_recon, stats = engine.compress_kv(past_kv, mode); comp_end = time.time()
                step_start = time.time(); outputs = model(input_ids[:, -1:], past_key_values=past_kv_recon, use_cache=True); torch.cuda.synchronize(); step_end = time.time()
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2); comp_latency = (comp_end - comp_start) * 1000; step_latency = (step_end - step_start) * 1000
            results[model_id][mode] = {"peak_vram_mb": peak_vram, "compression_latency_ms": comp_latency, "step_latency_ms": step_latency, "throughput_tok_per_sec": 1000.0 / step_latency if step_latency > 0 else 0, "compression_ratio": stats.get("ratio", 1.0)}
            print(f"    VRAM: {peak_vram:.2f} MB | Latency: {step_latency:.2f} ms")
        del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f: json.dump(results, f, indent=4)
    print(f"\nMemory metrics saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", nargs="+", default=["vram", "bandwidth", "latency", "throughput"])
    parser.add_argument("--models", nargs="+", default=["qwen2-1.5b"])
    parser.add_argument("--modes", nargs="+", default=["fp16", "rank8", "lcg"])
    parser.add_argument("--output", type=str, default="results/phase20/memory_metrics.json")
    args = parser.parse_args()
    measure_memory_metrics(args.models, args.modes, args.output)
