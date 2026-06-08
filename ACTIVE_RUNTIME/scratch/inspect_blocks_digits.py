import os
import sys
import torch
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

context_len = 4000
depth = 0.10

code = "351697"
needle = f"The special code is {code}."
question = "What is the special code? Answer in exactly the 6-digit code number."

needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)

def make_filler_text(tokenizer, target_tokens: int):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    num_repeats = (target_tokens // len(filler_tokens)) + 1
    all_filler_tokens = (filler_tokens * num_repeats)[:target_tokens]
    return tokenizer.decode(all_filler_tokens), len(all_filler_tokens)

target_filler = context_len - len(needle_tokens) - 100
filler_text, actual_filler_len = make_filler_text(tokenizer, max(10, target_filler))

split_char_idx = int(len(filler_text) * depth)
part1 = filler_text[:split_char_idx]
part2 = filler_text[split_char_idx:]

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    f"{part1}\n{needle}\n{part2}\n\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"Total prompt tokens: {len(prompt_ids)}")

# Block capacity calculation:
# prefill_len = len(prompt_ids)
prefill_len = len(prompt_ids)
raw_target = 64
target = min(raw_target, 256)
adaptive_size = max(16, ((target + 15) // 16) * 16)
block_capacity = 1 + adaptive_size

print(f"Adaptive block size: {adaptive_size}")
print(f"Block capacity: {block_capacity}")

# Scan blocks
for anchor_idx in range(0, len(prompt_ids), block_capacity):
    end = min(anchor_idx + block_capacity, len(prompt_ids))
    block_toks = prompt_ids[anchor_idx:end]
    decoded_tokens = [tokenizer.decode([tok_id]) for tok_id in block_toks]
    
    # Check if this block contains any digits
    has_digit = False
    digit_toks = []
    for tok_id, s in zip(block_toks, decoded_tokens):
        if any(c.isdigit() for c in s):
            has_digit = True
            digit_toks.append((tok_id, s))
            
    # Print blocks near the needle
    # Needle contains "special code is 351697"
    is_near_needle = any("special" in s or "code" in s or "351697" in s or "35" in s or "16" in s or "97" in s for s in decoded_tokens)
    
    if is_near_needle or has_digit:
        print(f"\n--- Block anchor_idx={anchor_idx} to {end} (Length: {end - anchor_idx}) ---")
        print(f"Exempted: {has_digit}")
        if has_digit:
            print(f"Digit tokens found: {digit_toks}")
        print("Tokens preview:")
        print(f"[{' | '.join(repr(s) for s in decoded_tokens[:15])} ... {' | '.join(repr(s) for s in decoded_tokens[-15:])}]")
