import os
import sys
import time
import mlx.core as mx

# Ensure root path is in sys.path
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

from mlx_lm.utils import load as mlx_load

def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model {model_id}...")
    model, tokenizer = mlx_load(model_id)
    
    # We will run native MLX (without our wrapper) first to see the speed
    prompt = "Apple Silicon unified memory is an architecture where"
    input_ids = mx.array([tokenizer.encode(prompt)])
    
    print("Running native MLX prefill...")
    t0 = time.perf_counter()
    logits = model(input_ids)
    mx.eval(logits)
    print(f"Prefill done in {(time.perf_counter() - t0)*1000:.1f}ms")
    
    # Run 10 decode steps
    print("\nRunning 10 native MLX decode steps...")
    token = mx.array([[123]]) # dummy token
    cache = model.make_cache()
    
    # Run one dummy step to warm up/compile cache
    model(token, cache=cache)
    
    for i in range(10):
        t0 = time.perf_counter()
        logits = model(token, cache=cache)
        mx.eval(logits)
        print(f"Step {i+1} done in {(time.perf_counter() - t0)*1000:.1f}ms")

if __name__ == "__main__":
    main()
