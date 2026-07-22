"""
tests/benchmark.py — Core serving benchmark.

Measures:
  - Prefill latency (s)
  - Decode throughput (tok/s)
  - Peak VRAM (GB)
  - Cosine similarity (compression quality)

Run with:
  DKV_MODEL=Qwen/Qwen2-7B-Instruct python tests/benchmark.py
"""
import time
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONTEXTS = [4096, 8192, 16384, 25000]
MAX_NEW_TOKENS = 128

def run_benchmark():
    from serving.hf_dkv_wrapper import DKVHFWrapper
    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    wrapper = DKVHFWrapper(MODEL, config={}, device="cuda")
    
    print(f"Model: {MODEL}")
    print(f"{'Context':>10} | {'Prefill(s)':>10} | {'Decode(tok/s)':>13} | {'PeakVRAM(GB)':>12} | {'AvgCosSim':>10}")
    print("-" * 65)
    
    for ctx in CONTEXTS:
        prompt = "word " * (ctx // 1)
        tokens_in = wrapper.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _ = wrapper.generate(prompt[:tokens_in * 4], max_new_tokens=1)
        prefill_t = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        _ = wrapper.generate(prompt[:tokens_in * 4], max_new_tokens=MAX_NEW_TOKENS)
        total_t = time.perf_counter() - t1
        decode_tps = MAX_NEW_TOKENS / max(total_t - prefill_t, 0.001)
        
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        summary = wrapper.manager.runtime_summary()
        cos_sim = summary.get("avg_cosine_sim", 0.0)
        
        print(f"{tokens_in:>10} | {prefill_t:>10.2f} | {decode_tps:>13.1f} | {peak_vram:>12.2f} | {cos_sim:>10.4f}")

if __name__ == "__main__":
    run_benchmark()
