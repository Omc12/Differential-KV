import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")
from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    with open("/Users/omchimurkar1/Desktop/Differential-KV/scratch/long_prompt.txt", "r") as f:
        prompt_content = f.read()
        
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + prompt_content + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Loading PyTorch DiffKV wrapper...")
    wrapper = PyTorchDiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 256},
        device=device,
    )
    
    print("Generating with PyTorch DiffKV...")
    # Set to exact or approximate attention based on env
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "0"
    
    out = wrapper.generate(prompt, max_new_tokens=50)
    print("\n--- PyTorch DiffKV Output ---")
    print(out)
    print("-----------------------------\n")
    
    wrapper.stop()

if __name__ == "__main__":
    main()
