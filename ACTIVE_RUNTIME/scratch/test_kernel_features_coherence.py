import os
import sys
import time
import psutil
import torch
import gc

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["DIFFKV_TELEMETRY"] = "1"
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "500"

def get_mps_mem():
    if hasattr(torch, "mps") and torch.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            return 0.0
    return 0.0

def get_cpu_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def run_test_for_attn_mode(approx_attn_value):
    print("=" * 70)
    print(f" TESTING WITH DIFFKV_MPS_APPROXIMATE_ATTN = {approx_attn_value}")
    print("=" * 70)
    
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = str(approx_attn_value)
    
    # Force fresh environment/model loading to prevent memory leak pollution between runs
    gc.collect()
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()
        
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Initial - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    # Load research paper text
    paper_path = os.path.join(_script_dir, "random_features_paper.txt")
    with open(paper_path, "r") as f:
        paper_text = f.read()
    
    # Load wrapper and model (use Rank=16)
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    print("Loading model wrapper...")
    t_start_load = time.time()
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"preset": "low", "rank": 16, "micro_block_size": 32},
        device=device
    )
    load_time = time.time() - t_start_load
    print(f"Model loaded in {load_time:.2f}s.")
    print(f"Post-load - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nRead the following research paper:\n\n"
        + paper_text +
        "\n\nQuestion: Based on the paper above, summarize the two proposed randomized feature maps, explain how they work conceptually/mathematically, and contrast their differences in detail. Do not repeat yourself. Be thorough.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    num_tokens = encoded.input_ids.shape[1]
    print(f"Prompt tokens: {num_tokens}")
    
    # Warmup / prefill and generate completely
    print("\nGenerating response completely (max 300 tokens)...")
    t0 = time.time()
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=300,
        temperature=0.0,  # Greedy decoding to check for deterministic repetition loops
        top_p=1.0,
        repetition_penalty=1.0, # Test natural repetition behavior of the model/caching system
    )
    elapsed = time.time() - t0
    
    print("\n" + "-" * 50)
    print("GENERATED RESPONSE:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    print(f"\nGeneration completed in {elapsed:.2f}s.")
    print(f"Post-gen - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    # Clean up resources
    wrapper.close()
    del wrapper
    gc.collect()
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()
    
    print(f"Post-cleanup - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB\n")

if __name__ == "__main__":
    # Test Exact attention mode (0) first, then Approximate attention mode (1)
    run_test_for_attn_mode(0)
    run_test_for_attn_mode(1)
