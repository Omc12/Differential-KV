import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"

import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def get_mps_memory_mb():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return 0.0

def benchmark_standard(context_lengths, num_decode_tokens=64):
    print("=" * 70)
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    
    mem_before_load = get_mps_memory_mb()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    mem_after_load = get_mps_memory_mb()
    model_weight_vram = mem_after_load - mem_before_load
    print(f"Standard Model loaded. Weights VRAM: {model_weight_vram:.2f} MB")
    
    results = {}
    for ctx_len in context_lengths:
        # Build prompt
        prompt = "word " * ctx_len
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        # We only take the exact context length
        input_ids = input_ids[:, :ctx_len]
        
        # Measure Prefill
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        mem_before_prefill = get_mps_memory_mb()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        t_prefill = time.perf_counter() - t0
        
        mem_after_prefill = get_mps_memory_mb()
        prefill_vram_overhead = mem_after_prefill - mem_before_prefill
        
        # Measure Decode
        past_key_values = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        
        t1 = time.perf_counter()
        current_past = past_key_values
        current_input = next_token_id
        
        with torch.no_grad():
            for i in range(num_decode_tokens):
                outputs = model(
                    input_ids=current_input,
                    past_key_values=current_past,
                    use_cache=True
                )
                current_past = outputs.past_key_values
                current_input = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                
        t_decode = time.perf_counter() - t1
        tps = num_decode_tokens / max(t_decode, 0.001)
        
        # Record VRAM at the end of decode
        mem_after_decode = get_mps_memory_mb()
        decode_vram_overhead = mem_after_decode - mem_before_prefill
        
        results[ctx_len] = {
            "prefill_s": t_prefill,
            "prefill_vram_mb": prefill_vram_overhead,
            "decode_tps": tps,
            "decode_vram_mb": decode_vram_overhead,
        }
        print(f"Standard Dense Context {ctx_len:5d} | Prefill: {t_prefill:.3f}s (VRAM: {prefill_vram_overhead:.1f}MB) | Decode: {tps:.1f} tok/s (VRAM: {decode_vram_overhead:.1f}MB)")

    del model
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return results

def benchmark_diffkv(context_lengths, num_decode_tokens=64):
    print("=" * 70)
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    # Configuration matches standard settings
    config = {
        "rank": 32,
        "micro_block_size": 256,
        "serving_mode": "balanced",
    }
    
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
        
    mem_before_load = get_mps_memory_mb()
    wrapper = DiffKVHFWrapper(
        model_id=MODEL_ID,
        config=config,
        device=DEVICE.type,
    )
    mem_after_load = get_mps_memory_mb()
    model_weight_vram = mem_after_load - mem_before_load
    print(f"DiffKV Model loaded. Weights VRAM: {model_weight_vram:.2f} MB")
    
    results = {}
    for ctx_len in context_lengths:
        prompt = "word " * ctx_len
        # Let's warm up tokenizer
        input_ids = wrapper.tokenizer(prompt, return_tensors="pt").input_ids[:, :ctx_len]
        trimmed_prompt = wrapper.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        mem_before_prefill = get_mps_memory_mb()
        
        # Measure Prefill (DiffKV generate handles chunked prefill + background compression)
        # We'll measure the full prefill stage
        t0 = time.perf_counter()
        
        # First generate 1 token to trigger the prefill pass and first compression barrier
        _ = wrapper.generate(prompt=trimmed_prompt, max_new_tokens=1, temperature=0.0)
        t_prefill = time.perf_counter() - t0
        
        mem_after_prefill = get_mps_memory_mb()
        prefill_vram_overhead = mem_after_prefill - mem_before_prefill
        
        # Clear session to measure decode properly
        wrapper.manager.clear_session("default")
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        
        # Measure Decode
        # We trigger the generation with context reusing
        # Wait, wrapper.generate does prefill + decode.
        # Let's measure decode by running generate for num_decode_tokens
        t1 = time.perf_counter()
        _ = wrapper.generate(prompt=trimmed_prompt, max_new_tokens=num_decode_tokens, temperature=0.0)
        t_total = time.perf_counter() - t1
        
        # Decode time is approximately total_time - prefill_time
        t_decode = t_total - t_prefill
        tps = num_decode_tokens / max(t_decode, 0.001)
        
        mem_after_decode = get_mps_memory_mb()
        decode_vram_overhead = mem_after_decode - mem_before_prefill
        
        # Get SVD metrics
        summary = wrapper.manager.runtime_summary()
        avg_cos_sim = summary.get("avg_cosine_sim", 0.0)
        compression_ratio = summary.get("compression_ratio", 1.0)
        
        results[ctx_len] = {
            "prefill_s": t_prefill,
            "prefill_vram_mb": prefill_vram_overhead,
            "decode_tps": tps,
            "decode_vram_mb": decode_vram_overhead,
            "avg_cos_sim": avg_cos_sim,
            "compression_ratio": compression_ratio,
        }
        print(f"DiffKV Context {ctx_len:5d} | Prefill: {t_prefill:.3f}s (VRAM: {prefill_vram_overhead:.1f}MB) | Decode: {tps:.1f} tok/s (VRAM: {decode_vram_overhead:.1f}MB) | CosSim: {avg_cos_sim:.4f} | Ratio: {compression_ratio:.1f}x")
        
        # Clear default session after run
        wrapper.manager.clear_session("default")

    wrapper.stop()
    del wrapper
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return results

def print_comparison_table(context_lengths, std_results, diff_results):
    print("\n" + "=" * 80)
    print("                       DIFFKV VS DENSE BASELINE BENCHMARK")
    print("=" * 80)
    print(f"{'Context':<8} | {'Mode':<8} | {'Prefill Time':<12} | {'Decode TPS':<12} | {'KV Cache VRAM':<15} | {'Quality':<12}")
    print("-" * 80)
    
    for ctx in context_lengths:
        std = std_results.get(ctx)
        diff = diff_results.get(ctx)
        
        # Standard rows
        if std is not None:
            print(f"{ctx:<8d} | {'Dense':<8} | {std['prefill_s']:>10.3f}s | {std['decode_tps']:>10.1f} | {std['decode_vram_mb']:>12.1f} MB | {'1.0000 (Exact)':<12}")
        else:
            print(f"{ctx:<8d} | {'Dense':<8} | {'Skipped':>11} | {'Skipped':>10} | {'Skipped (>11GB)':>15} | {'Too Heavy':<12}")
            
        # DiffKV rows
        if diff is not None:
            quality_str = f"{diff['avg_cos_sim']:.4f} Cos" if diff['avg_cos_sim'] > 0 else "N/A"
            print(f"{'':<8} | {'DiffKV':<8} | {diff['prefill_s']:>10.3f}s | {diff['decode_tps']:>10.1f} | {diff['decode_vram_mb']:>12.1f} MB | {quality_str:<12}")
        else:
            print(f"{'':<8} | {'DiffKV':<8} | {'N/A':>11} | {'N/A':>10} | {'N/A':>15} | {'N/A':<12}")
        print("-" * 80)

if __name__ == "__main__":
    all_context_lengths = [512, 1024, 2048, 4096, 8192]
    std_context_lengths = [512, 1024, 2048] # Keep under 2048 to prevent 11GB allocator caching
    diff_context_lengths = [512, 1024, 2048, 4096, 8192]
    
    print("Running Standard baseline benchmarks (up to 2048 context)...")
    std_res = benchmark_standard(std_context_lengths)
    
    print("\nRunning DiffKV benchmarks (up to 8192 context)...")
    diff_res = benchmark_diffkv(diff_context_lengths)
    
    print_comparison_table(all_context_lengths, std_res, diff_res)
