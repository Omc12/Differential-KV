import os
import sys
import time
import gc
import psutil
import torch

# Default to MLX unless forced to PyTorch
FORCE_PYTORCH = os.environ.get("DIFFKV_FORCE_PYTORCH", "0") == "1"

is_mlx = False
if not FORCE_PYTORCH:
    try:
        import mlx.core as mx
        is_mlx = True
    except ImportError:
        pass

# Configure environment variables BEFORE importing wrapper
os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "0"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure ACTIVE_RUNTIME is in PATH
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.hf_diffkv_wrapper import _trim_python_heap

def get_process_memory_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def get_gpu_memory_gb(device):
    if is_mlx:
        return mx.get_active_memory() / (1024 ** 3)
    else:
        if device == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
            return torch.mps.current_allocated_memory() / (1024 ** 3)
        elif device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0

def main():
    if is_mlx:
        MODEL = os.environ.get("DIFFKV_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    else:
        MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("      DIFFERENTIAL KV: EXACT SPARSE ATTENTION SCALE BENCHMARK (4K - 64K)")
    print("=" * 80)
    print(f"Backend: {'MLX (Python)' if is_mlx else 'PyTorch (MPS)'}")
    print(f"Device:  {device}")
    print(f"Model:   {MODEL}")
    print(f"Exact Attention: Enabled")
    print("=" * 80)

    # Load wrapper ONCE at startup to avoid loading memory peaks on every iteration
    wrapper = DiffKVHFWrapper(
        MODEL, 
        config={"rank": 16, "micro_block_size": 32, "serving_mode": "long-context"}, 
        device=device
    )

    # Let's warm up
    print("Warming up model...")
    warmup_prompt = "Warmup query " * 32
    wrapper.generate(warmup_prompt, max_new_tokens=5)
    print("Warmup complete.\n")

    # Define contexts to benchmark
    contexts = [4096, 8192, 16384, 32768, 64000]
    
    print(f"{'Context (Tok)':>13} | {'Prefill Time (s)':>18} | {'Decode (Tok/s)':>15} | {'GPU Mem (GB)':>14} | {'Process RAM (GB)':>18}")
    print("-" * 88)

    for ctx in contexts:
        # Clear the old session session cache completely
        wrapper.manager.clear_session("default")
        gc.collect()
        if is_mlx:
            mx.clear_cache()
        else:
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        _trim_python_heap()

        # Build prompt of specific length
        prompt = "The quick brown fox jumps over the lazy dog. " * (ctx // 10)
        tokens_in = len(wrapper.tokenizer(prompt).input_ids)
        
        # Measure prefill
        t0 = time.perf_counter()
        _ = wrapper.generate(prompt, max_new_tokens=1)
        prefill_time = time.perf_counter() - t0
        
        # Measure decode
        num_decode_tokens = 32
        t1 = time.perf_counter()
        _ = wrapper.generate(prompt, max_new_tokens=num_decode_tokens)
        total_time = time.perf_counter() - t1
        
        decode_time = max(total_time - prefill_time, 0.001)
        tps = num_decode_tokens / decode_time
        
        # Measure memory
        gpu_mem = get_gpu_memory_gb(device)
        process_ram = get_process_memory_gb()
        
        print(f"{tokens_in:>13} | {prefill_time:>18.2f} | {tps:>15.1f} | {gpu_mem:>14.2f} | {process_ram:>18.2f}")

    print("=" * 88)

if __name__ == "__main__":
    main()
