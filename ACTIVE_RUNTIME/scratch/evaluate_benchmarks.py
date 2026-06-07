import os
import sys
import time
import random
import torch
import argparse
from typing import List, Dict, Any, Tuple

# Set paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

def get_memory_gb() -> float:
    if torch.backends.mps.is_available():
        try:
            # Returns currently allocated MPS memory in bytes
            return torch.mps.current_allocated_memory() / 1e9
        except (AttributeError, Exception):
            pass
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def make_filler_text(tokenizer, target_tokens: int) -> Tuple[str, int]:
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

# ── 1. Needle In A Haystack (NIAH) ──────────────────────────────────────────
def run_niah(wrapper: DiffKVHFWrapper, context_len: int, depth: float) -> Dict[str, Any]:
    code = f"{random.randint(100000, 999999)}"
    needle = f"The special code is {code}."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    
    tokenizer = wrapper.tokenizer
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    
    target_filler = context_len - len(needle_tokens) - 100
    filler_text, actual_filler_len = make_filler_text(tokenizer, max(10, target_filler))
    
    # Place needle at specified depth
    split_char_idx = int(len(filler_text) * depth)
    part1 = filler_text[:split_char_idx]
    part2 = filler_text[split_char_idx:]
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{part1}\n{needle}\n{part2}\n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    t0 = time.perf_counter()
    mem_before = get_memory_gb()
    
    # Set generation limits
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.0
    )
    
    t1 = time.perf_counter()
    mem_after = get_memory_gb()
    
    # Strip prompt from response to check ONLY generated text
    ans = response
    if response.startswith(prompt):
        ans = response[len(prompt):]
    elif "assistant\n" in response:
        ans = response.split("assistant\n")[-1]
        
    ans_clean = ans.strip()
    success = code in ans_clean
    
    return {
        "success": success,
        "latency_sec": t1 - t0,
        "memory_gb": max(0.0, mem_after - mem_before),
        "expected": code,
        "got": ans_clean
    }

# ── 2. Multi-Needle In A Haystack ───────────────────────────────────────────
def run_multi_niah(wrapper: DiffKVHFWrapper, context_len: int, num_needles: int) -> Dict[str, Any]:
    codes = {}
    needles = []
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Jack",
             "Judy", "Mallory", "Niaj", "Oscar", "Peggy", "Rupert", "Sybil", "Trent", "Victor", "Walter"]
    
    selected_names = names[:num_needles]
    for name in selected_names:
        code = f"{random.randint(100000, 999999)}"
        codes[name] = code
        needles.append(f"The secret code for {name} is {code}.")
        
    tokenizer = wrapper.tokenizer
    needles_text = "\n".join(needles)
    needles_tokens = tokenizer.encode(needles_text + "\n", add_special_tokens=False)
    
    target_filler = context_len - len(needles_tokens) - 150
    filler_text, _ = make_filler_text(tokenizer, max(10, target_filler))
    
    # Distribute needles uniformly across the filler text
    parts = []
    chunk_size = len(filler_text) // (num_needles + 1)
    for i in range(num_needles):
        parts.append(filler_text[i*chunk_size : (i+1)*chunk_size])
        parts.append(needles[i])
    parts.append(filler_text[num_needles*chunk_size:])
    
    document = "\n".join(parts)
    names_query = ", ".join(selected_names)
    question = f"What are the secret codes for {names_query}? Respond strictly in format 'Name: Code', one per line."
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{document}\n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    t0 = time.perf_counter()
    response = wrapper.generate(prompt=prompt, max_new_tokens=128, temperature=0.0)
    t1 = time.perf_counter()
    
    ans = response
    if response.startswith(prompt):
        ans = response[len(prompt):]
    elif "assistant\n" in response:
        ans = response.split("assistant\n")[-1]
        
    ans_clean = ans.strip()
    
    # Evaluate matches
    matches = 0
    for name, code in codes.items():
        if f"{name}: {code}" in ans_clean or (name in ans_clean and code in ans_clean):
            matches += 1
            
    accuracy = (matches / num_needles) * 100
    
    return {
        "accuracy_pct": accuracy,
        "matches": matches,
        "total": num_needles,
        "latency_sec": t1 - t0,
        "got": ans_clean
    }

# ── 3. Lost In The Middle ───────────────────────────────────────────────────
def run_lost_in_the_middle(wrapper: DiffKVHFWrapper, context_len: int, position: str) -> Dict[str, Any]:
    # Position: 'beginning' (5%), 'middle' (50%), 'end' (95%)
    depth_map = {"beginning": 0.05, "middle": 0.50, "end": 0.95}
    depth = depth_map[position]
    
    # Re-use NIAH logic with exact depth
    res = run_niah(wrapper, context_len, depth)
    res["position"] = position
    return res

# ── 4. Cross-Chunk Reasoning (Multi-Hop) ───────────────────────────────────
def run_cross_chunk_reasoning(wrapper: DiffKVHFWrapper, context_len: int) -> Dict[str, Any]:
    # Put facts in different chunks
    # Chunk A (Early): "Alice is married to Bob."
    # Chunk B (Middle): "Bob's sister is Carol."
    # Chunk C (Late): "Carol lives in Paris."
    
    fact_a = "Alice is married to Bob."
    fact_b = "Bob's sister is Carol."
    fact_c = "Carol lives in Paris."
    
    tokenizer = wrapper.tokenizer
    facts_len = len(tokenizer.encode(fact_a + fact_b + fact_c, add_special_tokens=False))
    
    target_filler = context_len - facts_len - 150
    filler_text, _ = make_filler_text(tokenizer, max(10, target_filler))
    
    # Place at 15%, 50%, 85% depths
    len_f = len(filler_text)
    part1 = filler_text[:int(len_f * 0.15)]
    part2 = filler_text[int(len_f * 0.15):int(len_f * 0.50)]
    part3 = filler_text[int(len_f * 0.50):int(len_f * 0.85)]
    part4 = filler_text[int(len_f * 0.85):]
    
    question = "Where does the sister of Alice's husband live? Answer in exactly one word representing the city."
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{part1}\n{fact_a}\n{part2}\n{fact_b}\n{part3}\n{fact_c}\n{part4}\n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    t0 = time.perf_counter()
    response = wrapper.generate(prompt=prompt, max_new_tokens=16, temperature=0.0)
    t1 = time.perf_counter()
    
    ans = response
    if response.startswith(prompt):
        ans = response[len(prompt):]
    elif "assistant\n" in response:
        ans = response.split("assistant\n")[-1]
        
    ans_clean = ans.strip()
    success = "paris" in ans_clean.lower()
    
    return {
        "success": success,
        "latency_sec": t1 - t0,
        "got": ans_clean
    }

# ── 5. Contradiction / Update Test ──────────────────────────────────────────
def run_contradiction_test(wrapper: DiffKVHFWrapper, context_len: int) -> Dict[str, Any]:
    fact_old = "Alice currently lives in New York."
    fact_new = "Alice recently moved and currently lives in Seattle."
    
    tokenizer = wrapper.tokenizer
    facts_len = len(tokenizer.encode(fact_old + fact_new, add_special_tokens=False))
    
    target_filler = context_len - facts_len - 150
    filler_text, _ = make_filler_text(tokenizer, max(10, target_filler))
    
    # Place old fact at 20% and new fact at 80% depth
    len_f = len(filler_text)
    part1 = filler_text[:int(len_f * 0.20)]
    part2 = filler_text[int(len_f * 0.20):int(len_f * 0.80)]
    part3 = filler_text[int(len_f * 0.80):]
    
    question = "Where does Alice currently live? Answer in exactly one word representing the city."
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{part1}\n{fact_old}\n{part2}\n{fact_new}\n{part3}\n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    t0 = time.perf_counter()
    response = wrapper.generate(prompt=prompt, max_new_tokens=16, temperature=0.0)
    t1 = time.perf_counter()
    
    ans = response
    if response.startswith(prompt):
        ans = response[len(prompt):]
    elif "assistant\n" in response:
        ans = response.split("assistant\n")[-1]
        
    ans_clean = ans.strip()
    success = "seattle" in ans_clean.lower() and "new york" not in ans_clean.lower()
    
    return {
        "success": success,
        "latency_sec": t1 - t0,
        "got": ans_clean
    }


def main():
    parser = argparse.ArgumentParser(description="DiffKV Benchmark Suite")
    parser.add_argument("--full", action="store_true", help="Run the full grid of contexts and depths (takes longer)")
    args = parser.parse_args()
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print(f"Starting DiffKV Verification Benchmark Suite")
    print(f"Model:  {MODEL}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Configure SRL thresholds to be active for smaller contexts
    os.environ["DIFFKV_SRL_THRESHOLD"] = "5"
    
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    
    # Define matrix grid
    if args.full:
        contexts = [4000, 8000, 16000, 25000]
        depths = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        multi_needles = [5, 10, 20]
    else:
        # Fast smoke-test grid
        contexts = [4000, 8000]
        depths = [0.10, 0.50, 0.90]
        multi_needles = [5]
        
    print("\n--- Running 1. Needle In A Haystack (NIAH) ---")
    niah_results = []
    for ctx in contexts:
        for d in depths:
            print(f"Running NIAH: context={ctx}, depth={d*100:.0f}%...")
            res = run_niah(wrapper, ctx, d)
            niah_results.append((ctx, d, res))
            print(f"  Result: {'PASS' if res['success'] else 'FAIL'} | Got: {res['got']} (Expected: {res['expected']}) | Latency: {res['latency_sec']:.2f}s | Mem: {res['memory_gb']:.3f}GB")
            
    print("\n--- Running 2. Multi-Needle In A Haystack ---")
    multi_results = []
    for n_needles in multi_needles:
        ctx = 8000
        print(f"Running Multi-NIAH: context={ctx}, needles={n_needles}...")
        res = run_multi_niah(wrapper, ctx, n_needles)
        multi_results.append((n_needles, res))
        print(f"  Accuracy: {res['accuracy_pct']:.1f}% ({res['matches']}/{res['total']}) | Latency: {res['latency_sec']:.2f}s")
        print(f"  Got:\n{res['got']}")
        
    print("\n--- Running 3. Lost In The Middle ---")
    lim_results = []
    for pos in ["beginning", "middle", "end"]:
        ctx = 8000
        print(f"Running Lost-In-The-Middle: context={ctx}, position={pos}...")
        res = run_lost_in_the_middle(wrapper, ctx, pos)
        lim_results.append((pos, res))
        print(f"  Result: {'PASS' if res['success'] else 'FAIL'} | Got: {res['got']} | Latency: {res['latency_sec']:.2f}s")
        
    print("\n--- Running 4. Cross-Chunk Reasoning (Multi-Hop) ---")
    ctx = 8000
    print(f"Running Cross-Chunk Reasoning: context={ctx}...")
    cc_res = run_cross_chunk_reasoning(wrapper, ctx)
    print(f"  Result: {'PASS' if cc_res['success'] else 'FAIL'} | Got: {cc_res['got']} (Expected: Paris) | Latency: {cc_res['latency_sec']:.2f}s")
    
    print("\n--- Running 5. Contradiction / Update Test ---")
    print(f"Running Contradiction/Update: context={ctx}...")
    up_res = run_contradiction_test(wrapper, ctx)
    print(f"  Result: {'PASS' if up_res['success'] else 'FAIL'} | Got: {up_res['got']} (Expected: Seattle) | Latency: {up_res['latency_sec']:.2f}s")
    
    # Print beautiful summary report
    print("\n" + "=" * 60)
    print("                      BENCHMARK REPORT")
    print("=" * 60)
    
    print("\n1. Needle In A Haystack Grid (Accuracy % / Latency)")
    print(f"{'Context':<10} | " + " | ".join(f"D={d*100:.0f}%" for d in depths))
    print("-" * 60)
    # Pivot results by context length
    for ctx in contexts:
        row = f"{f'{int(ctx/1000)}K':<10} | "
        cols = []
        for d in depths:
            match = [r[2] for r in niah_results if r[0] == ctx and r[1] == d]
            if match:
                res = match[0]
                cols.append(f"{'100%' if res['success'] else '0%':<6} ({res['latency_sec']:.1f}s)")
            else:
                cols.append("N/A")
        print(row + " | ".join(cols))
        
    print("\n2. Multi-Needle In A Haystack")
    for n_needles, res in multi_results:
        print(f"  Needles: {n_needles:2d} | Accuracy: {res['accuracy_pct']:5.1f}% | Matches: {res['matches']}/{res['total']} | Latency: {res['latency_sec']:.1f}s")
        
    print("\n3. Lost In The Middle")
    for pos, res in lim_results:
        print(f"  Position: {pos:<10} | Accuracy: {'100%' if res['success'] else '0%':<5} | Latency: {res['latency_sec']:.1f}s")
        
    print("\n4. Sub-Component Tests")
    print(f"  Cross-Chunk Reasoning (Multi-Hop): {'PASS' if cc_res['success'] else 'FAIL'} ({cc_res['got']}) | Latency: {cc_res['latency_sec']:.2f}s")
    print(f"  Contradiction / Update Test:       {'PASS' if up_res['success'] else 'FAIL'} ({up_res['got']}) | Latency: {up_res['latency_sec']:.2f}s")
    print("=" * 60)
    
    wrapper.stop()

if __name__ == "__main__":
    main()
