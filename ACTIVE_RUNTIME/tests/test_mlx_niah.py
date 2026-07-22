"""
Phase 3 — MLX NIAH smoke test for the real DKV compressed decode path.

Tests the MLXDKVWrapper (active runtime) directly, with DKV_COMPRESSED_DECODE=1 (default).
The prompt is built to exactly the target context length and the needle is planted at a
specified depth. Success = the needle passcode appears in the generated output.

Run:
    cd ACTIVE_RUNTIME
    python tests/test_mlx_niah.py          # 4k context, depth 0.5
    python tests/test_mlx_niah.py 8000     # 8k context
    python tests/test_mlx_niah.py 4000 0.1 0.9  # 4k, multiple depths
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default: real compressed decode
os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")

NEEDLE = "The secret passcode is OMEGA-7741-DELTA."
QUESTION = "What is the secret passcode? Repeat it exactly."
FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)


def make_mlx_niah_prompt(tokenizer, target_tokens: int, depth: float) -> str:
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_toks = tokenizer.encode(NEEDLE + "\n", add_special_tokens=False)
    question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)
    system_overhead = 80  # approximate template tokens

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


def run_niah(ctx_len: int, depth: float, model_id: str) -> bool:
    from serving.mlx_dkv_wrapper import MLXDKVWrapper

    print(f"\n{'='*60}")
    print(f"MLX NIAH Test — ctx={ctx_len} tokens, depth={depth:.1f}")
    use_compressed = os.environ.get("DKV_COMPRESSED_DECODE", "1")
    print(f"DKV_COMPRESSED_DECODE={use_compressed}")
    print(f"{'='*60}")

    wrapper = MLXDKVWrapper(
        model_id=model_id,
        config={"rank": 16, "block_size": 256},
    )

    prompt = make_mlx_niah_prompt(wrapper.tokenizer, ctx_len, depth)
    prompt_toks = len(wrapper.tokenizer.encode(prompt))
    print(f"Prompt tokens: {prompt_toks}")

    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    # IMPORTANT: wrapper.generate() returns prompt + generation decoded together,
    # so the planted needle is always present in `response`. Checking the needle
    # against the full string is a false positive — extract ONLY the newly
    # generated tokens and check those.
    sid = wrapper.active_session or "default"
    all_ids = wrapper._session_token_ids.get(sid, [])
    gen_ids = all_ids[prompt_toks:]
    gen_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)
    wrapper.close()

    needle_recovered = "OMEGA-7741-DELTA" in gen_text
    status = "PASS ✓" if needle_recovered else "FAIL ✗"
    print(f"\nGenerated (new tokens only): {gen_text!r}")
    print(f"Needle in generation: {needle_recovered}")
    print(f"Result: {status}")
    return needle_recovered


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ctx", nargs="?", type=int, default=4000)
    parser.add_argument("depths", nargs="*", type=float, default=[0.5])
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    args = parser.parse_args()

    results = []
    for depth in args.depths:
        passed = run_niah(args.ctx, depth, args.model)
        results.append((args.ctx, depth, passed))

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for ctx, dep, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  ctx={ctx}  depth={dep:.1f}")
        if not ok:
            all_pass = False

    sys.exit(0 if all_pass else 1)
