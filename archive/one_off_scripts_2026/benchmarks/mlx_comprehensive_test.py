import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ACTIVE_RUNTIME"))

# Force compressed decode
os.environ["DKV_COMPRESSED_DECODE"] = "1"
# Ensure we set a default max residual of 128
os.environ.setdefault("DKV_MAX_RESIDUAL", "128")

from serving.mlx_dkv_wrapper import MLXDKVWrapper
import mlx.core as mx

NEEDLE = "The secret passcode is OMEGA-7741-DELTA."
QUESTION = "What is the secret passcode? Repeat it exactly."
FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

def make_prompt(tokenizer, target_tokens: int, depth: float) -> str:
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_toks = tokenizer.encode(NEEDLE + "\n", add_special_tokens=False)
    question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)
    system_overhead = 80

    filler_budget = target_tokens - len(needle_toks) - len(question_toks) - system_overhead
    if filler_budget < 0:
        filler_budget = 100

    repeats = (filler_budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:filler_budget]

    insert_at = int(len(all_filler) * depth)
    part1 = tokenizer.decode(all_filler[:insert_at])
    part2 = tokenizer.decode(all_filler[insert_at:])

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1 + "\n"
        + NEEDLE + "\n"
        + part2 + "\n\n"
        + QUESTION + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def run_test_case(wrapper, ctx_len: int, depth: float):
    prompt = make_prompt(wrapper.tokenizer, ctx_len, depth)
    prompt_toks = len(wrapper.tokenizer.encode(prompt))

    # Clean previous session state
    sid = "default"
    wrapper.manager.clear_session(sid)
    wrapper._session_token_ids[sid] = []

    # Run generation with timing
    t_start = time.perf_counter()
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    total_time = time.perf_counter() - t_start

    # Extract new tokens only
    all_ids = wrapper._session_token_ids.get(sid, [])
    gen_ids = all_ids[prompt_toks:]
    gen_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Verify accuracy
    needle_recovered = "OMEGA-7741-DELTA" in gen_text
    status = "PASS" if needle_recovered else "FAIL"

    # Analyze adaptive budgets selected for all compressed blocks
    session = wrapper.manager.sessions.get(sid)
    budget_stats = {8: 0, 16: 0, 128: 0, "other": 0}
    total_compressed_blocks = 0

    if session:
        for layer_idx in range(wrapper.manager.num_layers):
            nb = session["num_blocks"][layer_idx]
            if nb > 0:
                budgets = session["comp_res_n"][layer_idx][:nb]
                if hasattr(budgets, "tolist"):
                    budgets = budgets.tolist()
                for b in budgets:
                    # MLX arrays item conversion if needed
                    if hasattr(b, "item"):
                        b = int(b.item())
                    else:
                        b = int(b)
                    total_compressed_blocks += 1
                    if b == 8:
                        budget_stats[8] += 1
                    elif b == 16:
                        budget_stats[16] += 1
                    elif b == 128:
                        budget_stats[128] += 1
                    else:
                        budget_stats["other"] += 1

    # Print summary of the run
    print(f"\nContext: {ctx_len} | Depth: {depth:.1f} | Recovery: {status}")
    print(f"  Generated Output: {gen_text!r}")
    print(f"  Time: {total_time:.2f}s | Decode Speed: {len(gen_ids) / total_time:.1f} tok/s")
    print(f"  OPT-A Budget Distribution ({total_compressed_blocks} total blocks across all layers):")
    print(f"    - Capped at 8 (Easy Prose):    {budget_stats[8]} blocks")
    print(f"    - Capped at 16 (Medium):       {budget_stats[16]} blocks")
    print(f"    - Full 128 (Factual/Hard):     {budget_stats[128]} blocks")
    if budget_stats["other"] > 0:
        print(f"    - Other budget size:           {budget_stats['other']} blocks")

    return {
        "ctx": ctx_len,
        "depth": depth,
        "recovered": needle_recovered,
        "time": total_time,
        "tps": len(gen_ids) / total_time,
        "blocks_8": budget_stats[8],
        "blocks_16": budget_stats[16],
        "blocks_128": budget_stats[128],
    }

def main():
    model_id = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    print("Initializing MLXDKVWrapper...")
    wrapper = MLXDKVWrapper(
        model_id=model_id,
        config={"rank": 16, "block_size": 256},
    )

    test_cases = [
        # Context Length, Depth
        (4000, 0.1),
        (4000, 0.5),
        (4000, 0.9),
        (8000, 0.1),
        (8000, 0.5),
        (8000, 0.9),
        (16000, 0.1),
        (16000, 0.5),
        (16000, 0.9),
    ]

    results = []
    for ctx, depth in test_cases:
        res = run_test_case(wrapper, ctx, depth)
        results.append(res)

    print("\n" + "="*80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Ctx':>6} | {'Depth':>5} | {'Status':>6} | {'Time(s)':>8} | {'Speed(tok/s)':>12} | {'Budget:8':>8} | {'Budget:16':>9} | {'Budget:128':>10}")
    print("-" * 80)
    for r in results:
        status = "PASS" if r["recovered"] else "FAIL"
        print(f"{r['ctx']:>6} | {r['depth']:>5.1f} | {status:>6} | {r['time']:>8.2f} | {r['tps']:>12.1f} | {r['blocks_8']:>8} | {r['blocks_16']:>9} | {r['blocks_128']:>10}")
    print("="*80)

if __name__ == "__main__":
    main()
