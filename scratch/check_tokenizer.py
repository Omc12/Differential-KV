import os
import sys

# Add diffkv_native to path to import tokenizer
sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
hf_tokens = tokenizer.encode(prompt, add_special_tokens=False)
print("HF tokens (add_special_tokens=False):", hf_tokens)

hf_tokens_special = tokenizer.encode(prompt, add_special_tokens=True)
print("HF tokens (add_special_tokens=True) :", hf_tokens_special)

# Let's print the token representations for HF
for t in hf_tokens:
    print(f"  {t}: {tokenizer.decode([t])!r}")
