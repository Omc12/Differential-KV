import torch
import time
import json
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from gpu.qwen_patch_logic import allocation_aware_forward
from telemetry.strict_metric_taxonomy import StrictMetricTaxonomy
from telemetry.wallclock_enforcer import WallclockEnforcer

def apply_scientific_patch(model):
    """
    Physically monkey-patches the Qwen2Attention layers.
    """
    print("[PHASE 18.3A] Applying Physical Allocation-Aware Patch to all Attention Layers...")
    rotary_emb = model.model.rotary_emb
    for layer in model.model.layers:
        # We replace the forward method of the attention module
        # We also pass a reference to the rotary embeddings
        layer.self_attn._rotary_emb_ref = rotary_emb
        bound_method = allocation_aware_forward.__get__(layer.self_attn, layer.self_attn.__class__)
        layer.self_attn.forward = bound_method
    print("[SUCCESS] Model patched.")

def run_allocation_aware_benchmarks():
    print("="*60)
    print("PHASE 18.3 — ALLOCATION-AWARE SPARSE ATTENTION (PATCHED)")
    print("="*60)

    taxonomy = StrictMetricTaxonomy()
    enforcer = WallclockEnforcer()

    # 1. Load and Patch
    loader = Qwen7BRealLoader()
    model = loader.load()
    apply_scientific_patch(model)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # 2. Context Test Matrix
    # We use 512 chunks for input to minimize peak allocation
    contexts = [4096, 8192, 16384]
    results = []

    model.eval()

    for ctx_len in contexts:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"\n[RUN] Context: {ctx_len} (Optimized Patched AASAE)")
        
        # Prepare Input
        text = "AASAE optimization bypasses O(n^2) bottlenecks." * (ctx_len // 10)
        prompt_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len).input_ids.to("cuda")

        initial_vram = torch.cuda.memory_allocated() / (1024**3)

        try:
            enforcer.start()
            
            # Use DynamicCache for explicit management
            from transformers import DynamicCache
            past_key_values = DynamicCache()
            chunk_size = 512
            
            peak_vram = initial_vram
            
            for i in range(0, ctx_len, chunk_size):
                chunk = prompt_ids[:, i:i+chunk_size]
                with torch.no_grad():
                    outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
                    # DynamicCache is updated in-place
                    peak_vram = max(peak_vram, torch.cuda.memory_allocated() / (1024**3))
            
            # Measure decode TPS
            generated_tokens = 0
            curr_input = prompt_ids[:, -1:]
            for _ in range(16):
                outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                curr_input = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
                generated_tokens += 1
                
            duration = enforcer.stop()
            avg_tps = generated_tokens / duration
            
            print(f"[RESULT] {taxonomy.log_measured('TPS', avg_tps)}")
            print(f"[RESULT] {taxonomy.log_measured('Peak VRAM', peak_vram, 'GB')}")

            results.append({
                "context": ctx_len,
                "tps": avg_tps,
                "peak_vram": peak_vram,
                "status": "SUCCESS"
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAILURE] Context {ctx_len} failed: {e}")
            results.append({
                "context": ctx_len,
                "status": "FAILED",
                "error": str(e)
            })
            torch.cuda.empty_cache()

    # 3. Export Results
    export_path = "results/reconstruction_18_3/bench_results.json"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    with open(export_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_allocation_aware_benchmarks()
