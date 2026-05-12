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

def run_throughput_eval(batch_sizes, context_lengths, output_path):
    results = {}
    model_id = "Qwen/Qwen2-1.5B"
    print(f"\n>>> Running Throughput Eval on {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        engine = UniversalCompressionEngine(model, tokenizer)
    except Exception as e:
        print(f"Error loading {model_id}: {e}"); return
    for bs in batch_sizes:
        results[bs] = {}
        for ctx_raw in context_lengths:
            ctx_len = int(ctx_raw.replace("k", "000").replace("K", "000"))
            print(f"  Batch Size: {bs} | Context Length: {ctx_len}")
            input_ids = torch.randint(0, 1000, (bs, ctx_len), device=model.device)
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True)
                past_kv_recon, _ = engine.compress_kv(outputs.past_key_values, "lcg")
                latencies, curr_ids, curr_kv = [], torch.randint(0, 1000, (bs, 1), device=model.device), past_kv_recon
                for _ in range(5):
                    torch.cuda.synchronize(); start = time.time()
                    outputs = model(curr_ids, past_key_values=curr_kv, use_cache=True)
                    torch.cuda.synchronize(); end = time.time(); latencies.append((end - start) * 1000)
                    curr_ids, curr_kv = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True), outputs.past_key_values
                avg_latency = sum(latencies) / len(latencies)
                results[bs][ctx_raw] = {"avg_latency_ms": avg_latency, "throughput_tok_per_sec": (bs * 1000.0) / avg_latency if avg_latency > 0 else 0, "vram_gb": torch.cuda.memory_allocated() / (1024**3)}
        del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f: json.dump(results, f, indent=4)
    print(f"\nThroughput metrics saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--context_lengths", nargs="+", default=["4k", "8k", "16k", "32k"])
    parser.add_argument("--output", type=str, default="results/phase20/throughput_metrics.json")
    args = parser.parse_args()
    run_throughput_eval(args.batch_sizes, args.context_lengths, args.output)
