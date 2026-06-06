import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_correctness():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Loading model...")
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    
    print("\n--- Generating response ---")
    response = wrapper.generate(prompt, max_new_tokens=20)
    print(f"Generated: {repr(response)}")

if __name__ == "__main__":
    test_correctness()
