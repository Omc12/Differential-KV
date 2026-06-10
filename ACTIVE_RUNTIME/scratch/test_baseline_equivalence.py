import os
import sys
import torch
import numpy as np
import mlx.core as mx

# Ensure root path is in sys.path
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm.utils import load as mlx_load

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Apple Silicon unified memory is an architecture where"
    
    print("1. Loading HF model on MPS...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="mps"
    )
    
    print("2. Loading MLX model...")
    mlx_model, mlx_tokenizer = mlx_load(model_id)
    
    print("3. running HF...")
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
    hf_logits = hf_outputs.logits.cpu().numpy()
    
    print("4. running MLX...")
    mlx_inputs = mx.array(tokenizer(prompt, return_tensors="pt").input_ids.numpy())
    mlx_outputs = mlx_model(mlx_inputs)
    mx.eval(mlx_outputs)
    mlx_logits = np.array(mlx_outputs.astype(mx.float32))
    
    # Compare
    print("\nLogits Comparison:")
    print("HF shape:", hf_logits.shape)
    print("MLX shape:", mlx_logits.shape)
    
    hf_last = hf_logits[:, -1, :]
    mlx_last = mlx_logits[:, -1, :]
    
    mae = np.mean(np.abs(hf_last - mlx_last))
    max_err = np.max(np.abs(hf_last - mlx_last))
    print(f"MAE: {mae:.6f}")
    print(f"Max Diff: {max_err:.6f}")
    
    # Generate HF
    print("\nHF Greedy generation:")
    hf_gen_ids = hf_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    print("HF Output:", tokenizer.decode(hf_gen_ids[0]))
    
    # Generate MLX
    print("\nMLX Greedy generation:")
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    mlx_gen = mlx_generate(mlx_model, mlx_tokenizer, prompt=prompt, max_tokens=32, sampler=make_sampler(temp=0.0))
    print("MLX Output:", prompt + " " + mlx_gen)

if __name__ == "__main__":
    main()
