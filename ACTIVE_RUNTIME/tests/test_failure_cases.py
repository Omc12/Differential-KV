import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def make_filler(tokenizer, target_tokens):
    filler_unit = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    tokens = tokenizer.encode(filler_unit, add_special_tokens=False)
    num_repeats = (target_tokens // len(tokens)) + 1
    return tokenizer.decode((tokens * num_repeats)[:target_tokens])

def test_dense_unique_facts():
    """
    Scenario 1: Dense unique facts (reconstruction degradation stress).
    Loads a sequence containing multiple unique facts and asks about one.
    """
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set threshold low so routing is active
    os.environ["DIFFKV_SRL_THRESHOLD"] = "5"
    
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    # Create a dense list of facts
    facts = [
        "Alice lives in Boston.",
        "Bob lives in Seattle.",
        "Charlie lives in Chicago.",
        "David lives in Miami.",
        "Eve lives in Austin.",
        "Frank lives in Denver.",
        "Grace lives in Phoenix.",
        "Henry lives in Portland.",
        "Ivy lives in Atlanta.",
        "Jack lives in Detroit."
    ]
    
    # Add filler to push facts into compressed history
    filler = make_filler(wrapper.tokenizer, 3000)
    
    # Structure prompt
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + "\n".join(facts) + "\n"
        + filler + "\n\n"
        "Question: Where does David live? Answer in exactly one city name.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("\nRunning Dense Unique Facts test...")
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    print(f"Response: {response!r}")
    assert "Miami" in response, f"Failed dense unique facts retrieval. Output: {response}"
    wrapper.stop()

def test_far_needle_retrieval():
    """
    Scenario 2: Far-needle retrieval (0% depth, max recency window bypass).
    Needle is placed at the absolute start of the context (index 0).
    """
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set threshold low so routing is active
    os.environ["DIFFKV_SRL_THRESHOLD"] = "5"
    
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    needle = "The special security word is BANANA."
    filler = make_filler(wrapper.tokenizer, 4000)
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + needle + "\n"
        + filler + "\n\n"
        "Question: What is the special security word? Answer in exactly one word.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("\nRunning Far Needle Retrieval test (0% depth)...")
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    print(f"Response: {response!r}")
    assert "BANANA" in response, f"Failed far needle retrieval. Output: {response}"
    wrapper.stop()

def test_multi_hop_cross_reference():
    """
    Scenario 3: Multi-hop cross-referencing.
    Requires joining two facts separated by large filler contexts.
    """
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set threshold low so routing is active
    os.environ["DIFFKV_SRL_THRESHOLD"] = "5"
    
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    fact_a = "Alice is married to David."
    filler_a = make_filler(wrapper.tokenizer, 2000)
    fact_b = "David works in Seattle."
    filler_b = make_filler(wrapper.tokenizer, 2000)
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + fact_a + "\n"
        + filler_a + "\n"
        + fact_b + "\n"
        + filler_b + "\n\n"
        "Question: Where does Alice's husband work? Answer in exactly one city name.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("\nRunning Multi-hop Cross-reference test...")
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    print(f"Response: {response!r}")
    assert "Seattle" in response, f"Failed multi-hop cross-reference. Output: {response}"
    wrapper.stop()
