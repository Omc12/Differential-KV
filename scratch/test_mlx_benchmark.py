import os
import sys
import torch

# Ensure ACTIVE_RUNTIME is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. "
    )
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    
    target_filler_tokens = context_length - len(needle_tokens) - 100
    if target_filler_tokens < 0:
        target_filler_tokens = 100
        
    num_repeats = (target_filler_tokens // len(filler_tokens)) + 1
    all_filler_tokens = (filler_tokens * num_repeats)[:target_filler_tokens]
    
    insert_idx = int(len(all_filler_tokens) * depth)
    part1_tokens = all_filler_tokens[:insert_idx]
    part2_tokens = all_filler_tokens[insert_idx:]
    
    part1_text = tokenizer.decode(part1_tokens)
    part2_text = tokenizer.decode(part2_tokens)
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1_text + "\n"
        + needle + "\n"
        + part2_text + "\n\n"
        + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def test():
    # Force preset low so it runs easily
    config = {
        "preset": "low",
        "rank": 16,
    }
    
    print("Initializing MLX DiffKV wrapper...")
    wrapper = DiffKVHFWrapper(
        model_id=MODEL_ID,
        config=config,
        device="mps"
    )
    
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    
    prompt = make_niah_prompt(wrapper.tokenizer, 1024, 0.5, needle, question)
    
    print("Generating response...")
    response_full = wrapper.generate(prompt=prompt, max_new_tokens=64, temperature=0.0)
    
    prompt_len = len(wrapper.tokenizer(prompt).input_ids)
    session_id = wrapper.active_session or "default"
    stored_ids = getattr(wrapper, "_session_token_ids", {}).get(session_id, [])
    new_tokens = stored_ids[prompt_len:]
    response = wrapper.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    print(f"Full response: {repr(response_full[:200])}...")
    print(f"Generated suffix: {repr(response)}")
    print(f"Accuracy match: {'847291' in response}")
    
    wrapper.stop()

if __name__ == "__main__":
    test()
