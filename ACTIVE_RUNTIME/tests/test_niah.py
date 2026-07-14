import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    
    # Estimate remaining room for templates and questions
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

@pytest.mark.parametrize("depth", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("context_len", [4000, 8000])
def test_niah_depths(depth, context_len):
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set threshold low so SRL routing is active for these block counts
    os.environ["DIFFKV_SRL_THRESHOLD"] = "5"
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_SRL_VERBOSE"] = "1"
    
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    
    prompt = make_niah_prompt(wrapper.tokenizer, context_len, depth, needle, question)
    
    print(f"\nRunning NIAH test: context={context_len}, depth={depth}")
    
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    
    print(f"Response: {response!r}")
    
    # Check if correct code is present in generated tokens only (avoid false positive from prompt)
    prompt_toks = len(wrapper.tokenizer.encode(prompt))
    sid = wrapper.active_session or "default"
    all_ids = wrapper._session_token_ids.get(sid, [])
    gen_ids = all_ids[prompt_toks:]
    gen_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    print(f"Generated text only: {gen_text!r}")
    
    assert "847291" in gen_text, f"Failed to retrieve needle '847291' at context_len={context_len}, depth={depth}. Generated: {gen_text!r}"
    wrapper.stop()
