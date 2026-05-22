import torch
import time
import os
import json
from typing import List, Dict, Any
from inference.real_model_loader import RealModelLoader
from optimization.low_overhead_decode_loop import LowOverheadDecodeLoop
from inference.real_kv_cache_connector import RealKVCacheConnector

def run_extreme_context_benchmark(
    model_id: str,
    context_lengths: List[int] = [1024, 8192, 32768, 131072, 262144],
    gen_length: int = 128,
    sparse_mode: bool = True
):
    """
    PHASE 11C: LONG-CONTEXT SPARSE ADVANTAGE VALIDATION
    
    Benchmarks real transformer generation at extreme context lengths.
    """
    print(f"--- EXTREME CONTEXT GENERATION BENCHMARK (Sparse: {sparse_mode}) ---")
    
    loader = RealModelLoader()
    model, tokenizer = loader.load(model_id)
    
    # Mock manager for now, in a real run this would be the actual DiffKV manager
    from runtime.kv_runtime_manager import KVRuntimeManager
    config = {"block_size": 64, "mode": "lowrank_sparse" if sparse_mode else "fp16"}
    manager = KVRuntimeManager(config, device="cuda")
    connector = RealKVCacheConnector(manager)
    
    # Add update_fast and get_last_overhead to connector for compatibility with LowOverheadDecodeLoop
    if not hasattr(connector, 'update_fast'):
        connector.update_fast = connector.update
    if not hasattr(connector, 'get_last_overhead'):
        connector.get_last_overhead = lambda: 0.0
    
    loop = LowOverheadDecodeLoop(model, tokenizer, connector)
    
    results = []
    
    for ctx_len in context_lengths:
        print(f"Testing context length: {ctx_len}...")
        
        # Create a real prompt of the specified length
        prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx_len), device="cuda")
        
        try:
            # Warmup
            _ = loop.decode(prompt_ids[:, :128], max_new_tokens=5)
            
            # Real run
            output = loop.decode(prompt_ids, max_new_tokens=gen_length)
            
            result = {
                "context_length": ctx_len,
                "gen_length": gen_length,
                "latency": output["latency"],
                "tps": output["tokens_per_sec"],
                "sparse_mode": sparse_mode,
                "model": model_id
            }
            results.append(result)
            print(f"  TPS: {result['tps']:.2f}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at context length {ctx_len}")
            results.append({"context_length": ctx_len, "error": "OOM"})
            break
            
    # Save results
    output_dir = "results/reconstruction_11"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/extreme_context_{'sparse' if sparse_mode else 'dense'}.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    # Example usage
    run_extreme_context_benchmark("Qwen/Qwen2-7B-Instruct", context_lengths=[1024, 4096])
