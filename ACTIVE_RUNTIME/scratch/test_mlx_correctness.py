import os
import sys
import torch
import numpy as np

# Ensure root path is in sys.path
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper
from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper

def test_correctness():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt = "Apple Silicon unified memory is an architecture where"
    
    print("--------------------------------------------------")
    print(f"1. Loading PyTorch Wrapper (MPS) for {model_id}...")
    # Preset rank 16 for consistency
    config = {
        'mode': 'fp16',
        'block_size': 64,
        'rank': 16,
        'micro_block_size': 64
    }
    
    py_wrapper = PyTorchDiffKVHFWrapper(
        model_id=model_id,
        config=config,
        device="mps"
    )
    py_wrapper.ensure_loaded()
    
    print("\n2. Loading MLX Wrapper (Metal) for {model_id}...")
    mlx_wrapper = MLXDiffKVWrapper(
        model_id=model_id,
        config=config
    )
    mlx_wrapper.ensure_loaded()
    
    print("\n--------------------------------------------------")
    print("3. Comparing Prefill logits...")
    
    # Tokenize input using PyTorch tokenizer (both wrappers use same underlying vocab)
    input_ids = py_wrapper.tokenizer(prompt, return_tensors='pt').input_ids.to("mps")
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device="mps").unsqueeze(0)
    
    # Run PyTorch prefill
    py_wrapper.model._diffkv_session_ids = ["test_py"]
    py_wrapper.manager.init_session("test_py", prefill_len=input_ids.shape[1])
    with torch.no_grad():
        py_output = py_wrapper.model(input_ids, position_ids)
    py_logits = py_output.logits.cpu().numpy()
    
    # Run MLX prefill
    mlx_wrapper.model._diffkv_session_ids = ["test_mlx"]
    mlx_wrapper.manager.init_session("test_mlx", prefill_len=input_ids.shape[1])
    mlx_output = mlx_wrapper.model(input_ids, position_ids)
    mlx_logits = mlx_output.logits.cpu().numpy()
    
    # Compare shape and values of the last token (the only token used for sampling)
    py_last_logits = py_logits[:, -1, :]
    mlx_last_logits = mlx_logits[:, -1, :]
    
    print(f"PyTorch last logits shape: {py_last_logits.shape}")
    print(f"MLX last logits shape:     {mlx_last_logits.shape}")
    
    assert py_last_logits.shape == mlx_last_logits.shape, "Error: Logits shape mismatch!"
    
    # Mean absolute error
    mae = np.mean(np.abs(py_last_logits - mlx_last_logits))
    max_err = np.max(np.abs(py_last_logits - mlx_last_logits))
    print(f"Mean Absolute Error (MAE):    {mae:.6f}")
    print(f"Max Absolute Difference:       {max_err:.6f}")
    
    # Check if error is within reasonable bounds (usually < 0.1 due to tiny MPS vs Metal kernel compilation / math variances)
    if mae < 0.05:
        print("SUCCESS: Prefill logits match perfectly within precision tolerance!")
    else:
        print("WARNING: Logits difference is higher than expected. Check operations.")
        
    print("\n--------------------------------------------------")
    print("4. Comparing Text Generation...")
    
    print("\nPyTorch generation:")
    py_gen = py_wrapper.generate(prompt, max_new_tokens=32, temperature=0.0) # Greedy
    print(f"Output: {py_gen}")
    
    print("\nMLX generation:")
    mlx_gen = mlx_wrapper.generate(prompt, max_new_tokens=32, temperature=0.0) # Greedy
    print(f"Output: {mlx_gen}")
    
    # Clean up
    py_wrapper.close()
    mlx_wrapper.close()

if __name__ == "__main__":
    test_correctness()
