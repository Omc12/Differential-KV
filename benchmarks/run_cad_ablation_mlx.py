#!/usr/bin/env python3
"""
run_cad_ablation_mlx.py — Content-Aware residual selection ablation (MLX)
==========================================================================
Ablation study comparing three residual-selection policies on the same DKV
compressed model, at the same memory budget (max_residual, rank, etc.):

  baseline   — raw L2/joint reconstruction error only (no content boosts)
               DKV_RESIDUAL_TOKEN_BOOST=0
  shape_only — core-segment, owner, table capture ON; rarity OFF
               DKV_RESIDUAL_TOKEN_BOOST=8, DKV_RESIDUAL_RARITY_CAPTURE=0
  full_cad   — all captures ON (default)
               DKV_RESIDUAL_TOKEN_BOOST=8, DKV_RESIDUAL_RARITY_CAPTURE=1

Tasks evaluated (all implemented inline, no external dataset download):
  ruler_kv         — exact numeric/code retrieval (NIAH-style, 6-digit code)
  single_niah      — single passcode in haystack
  multi_niah       — two-needle multikey
  multihop_qa      — two-hop link (key→value chain)
  variable_tracking — LISP-style assignment chain

Usage:
  python benchmarks/run_cad_ablation_mlx.py [--model MODEL_ID]
         [--contexts 4096 8192] [--num-samples 10]
         [--tasks ruler_kv single_niah multi_niah multihop_qa variable_tracking]
         [--output results/cad_ablation.json]
         [--conditions baseline shape_only full_cad]
         [--seed 42]

Quick smoke test (CPU-safe tokenizer check only, no model inference):
  python benchmarks/run_cad_ablation_mlx.py --smoke-test
"""

import sys
import os
import re
import json
import time
import random
import string
import argparse
import gc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Filler corpus ──────────────────────────────────────────────────────────────
FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
    "Transformers revolutionized natural language processing beginning in 2017. "
    "Large language models have demonstrated emergent capabilities at scale. "
    "Researchers continue to investigate the limits of in-context learning. "
    "Memory and context length remain central challenges for efficient inference. "
)

# ── Condition definitions ──────────────────────────────────────────────────────
CONDITIONS = {
    "baseline": {
        "DKV_COMPRESSED_DECODE": "1",
        "DKV_RESIDUAL_TOKEN_BOOST": "0",     # raw L2 only
        "DKV_RESIDUAL_RARITY_CAPTURE": "0",
        "DKV_RESIDUAL_OWNER_CAPTURE": "0",
        "DKV_RESIDUAL_TABLE_CAPTURE": "0",
        "DKV_MAX_RESIDUAL": "128",
        "DKV_SPARSE_PREFILL": "1",
        "DKV_DECODE_CACHE": "1",
        "DKV_SPARSE_BIAS": "auto",
    },
    "shape_only": {
        "DKV_COMPRESSED_DECODE": "1",
        "DKV_RESIDUAL_TOKEN_BOOST": "8",
        "DKV_RESIDUAL_RARITY_CAPTURE": "0",   # rarity OFF
        "DKV_RESIDUAL_OWNER_CAPTURE": "1",
        "DKV_RESIDUAL_TABLE_CAPTURE": "1",
        "DKV_MAX_RESIDUAL": "128",
        "DKV_SPARSE_PREFILL": "1",
        "DKV_DECODE_CACHE": "1",
        "DKV_SPARSE_BIAS": "auto",
    },
    "full_cad": {
        "DKV_COMPRESSED_DECODE": "1",
        "DKV_RESIDUAL_TOKEN_BOOST": "8",
        "DKV_RESIDUAL_RARITY_CAPTURE": "1",   # rarity ON (new)
        "DKV_RESIDUAL_OWNER_CAPTURE": "1",
        "DKV_RESIDUAL_TABLE_CAPTURE": "1",
        "DKV_MAX_RESIDUAL": "128",
        "DKV_SPARSE_PREFILL": "1",
        "DKV_DECODE_CACHE": "1",
        "DKV_SPARSE_BIAS": "auto",
    },
    "dense": {
        "DKV_COMPRESSED_DECODE": "0",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def rand_code(rng, length=12):
    p1 = "".join(rng.choices(string.ascii_uppercase, k=5))
    p2 = "".join(rng.choices(string.digits, k=4))
    p3 = "".join(rng.choices(string.ascii_uppercase, k=4))
    return f"{p1}-{p2}-{p3}"


def pad_to_tokens(tokenizer, text_parts, target_tokens, rng):
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    parts_toks = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in text_parts)
    budget = max(50, target_tokens - parts_toks - 80)
    reps = (budget // len(filler_toks)) + 2
    all_filler = (filler_toks * reps)[:budget]
    filler_text = tokenizer.decode(all_filler)
    n = len(text_parts)
    chunk = max(1, len(filler_text) // (n + 1))
    out = []
    for i, part in enumerate(text_parts):
        out.append(filler_text[i * chunk:(i + 1) * chunk])
        out.append(part)
    out.append(filler_text[n * chunk:])
    return "".join(out)


def qwen_prompt(system, user):
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Task builders ──────────────────────────────────────────────────────────────

def build_ruler_kv(tokenizer, target_tokens, rng):
    """ruler_kv: exact numeric passkey buried deep in structured JSON-like filler.
    The filler uses colons, braces and brackets to stress-test the residual
    selector: without content-aware selection these structural tokens consume
    all residual slots, evicting the passkey digits."""
    code = "".join(rng.choices(string.digits, k=6))
    # JSON-like filler that contains lots of punctuation to stress the selector
    json_filler = (
        '{"session_id": "X", "metadata": {"version": 3, "flags": [0, 1, 0]}, '
        '"events": [{"type": "start", "ts": 1720000000}, '
        '{"type": "end", "ts": 1720000042}], "status": "ok"} '
    )
    needle = f'\n"secret_key": "{code}"\n'
    question = f'What is the value of "secret_key" in the data above? State only the value.'
    body = pad_to_tokens(tokenizer, [json_filler * 3 + needle], target_tokens, rng)
    prompt = qwen_prompt("You are a data extraction assistant.", body + "\n\n" + question)
    return {"task": "ruler_kv", "answer": code, "prompt": prompt, "check": "exact_in"}


def build_single_niah(tokenizer, target_tokens, rng):
    """single_niah: one alphanumeric passcode buried in prose filler."""
    code = rand_code(rng)
    needle = f"\nThe secret passcode is {code}.\n"
    question = "What is the secret passcode? Repeat it exactly, nothing else."
    body = pad_to_tokens(tokenizer, [needle], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "single_niah", "answer": code, "prompt": prompt, "check": "exact_in"}


def build_multi_niah(tokenizer, target_tokens, rng):
    """multi_niah: two needles; question specifies which key to retrieve."""
    key_a, code_a = rand_code(rng), rand_code(rng)
    key_b, code_b = rand_code(rng), rand_code(rng)
    needle_a = f"\nThe passcode for key {key_a} is {code_a}.\n"
    needle_b = f"\nThe passcode for key {key_b} is {code_b}.\n"
    question = f"What is the passcode for key {key_b}? State only the passcode."
    body = pad_to_tokens(tokenizer, [needle_a, needle_b], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "multi_niah", "answer": code_b, "prompt": prompt, "check": "exact_in"}


def build_multihop_qa(tokenizer, target_tokens, rng):
    """multihop_qa: two-hop retrieval chain (classification_key → passcode)."""
    classification_key = rand_code(rng)
    project_name = "Project " + "".join(rng.choices(string.ascii_uppercase, k=5))
    passcode = rand_code(rng)
    needle_a = f"\nThe security classification key for {project_name} is {classification_key}.\n"
    needle_b = f"\nThe passcode for {project_name} is {passcode}.\n"
    question = (
        f"What is the passcode for the project whose security classification key "
        f"is {classification_key}? State only the passcode."
    )
    body = pad_to_tokens(tokenizer, [needle_a, needle_b], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "multihop_qa", "answer": passcode, "prompt": prompt, "check": "exact_in"}


def build_variable_tracking(tokenizer, target_tokens, rng):
    """variable_tracking: LISP-style assignment chain; find terminal value."""
    depth = 8
    final_val = rand_code(rng)
    chain = [f"Let x0 = {final_val}."]
    for i in range(1, depth):
        chain.append(f"Let x{i} = x{i - 1}.")
    chain_text = " ".join(chain)
    question = f"What is the value of x{depth - 1}? State only the value."
    chain_part = "\n" + chain_text + "\n"
    body = pad_to_tokens(tokenizer, [chain_part], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "variable_tracking", "answer": final_val, "prompt": prompt, "check": "exact_in"}


TASK_BUILDERS = {
    "ruler_kv":          build_ruler_kv,
    "single_niah":       build_single_niah,
    "multi_niah":        build_multi_niah,
    "multihop_qa":       build_multihop_qa,
    "variable_tracking": build_variable_tracking,
}


# ── Scoring ────────────────────────────────────────────────────────────────────
def score(prediction, answer, check):
    pred = prediction.strip()
    if check == "exact_in":
        return 1.0 if answer in pred else 0.0
    return 0.0


# ── Main runner ────────────────────────────────────────────────────────────────
def run_cad_ablation(
    model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    contexts=None,
    num_samples=10,
    tasks=None,
    conditions=None,
    output_path=None,
    seed=42,
):
    if contexts is None:
        contexts = [4096, 8192]
    if tasks is None:
        tasks = list(TASK_BUILDERS.keys())
    if conditions is None:
        conditions = ["baseline", "shape_only", "full_cad"]

    print("Content-Aware Decision (CAD) Ablation — MLX", flush=True)
    print(f"Model:      {model_id}", flush=True)
    print(f"Contexts:   {contexts}", flush=True)
    print(f"Tasks:      {tasks}", flush=True)
    print(f"Conditions: {conditions}", flush=True)
    print(f"Samples:    {num_samples} per (task × context × condition)\n", flush=True)

    try:
        import mlx.core as mx
        from serving.mlx_dkv_wrapper import MLXDKVWrapper
    except ImportError as e:
        print(f"ERROR: cannot import MLX stack: {e}", flush=True)
        sys.exit(1)

    wrapper = MLXDKVWrapper(model_id=model_id, config={"rank": 32, "block_size": 1024})
    wrapper.ensure_loaded()
    tokenizer = wrapper.tokenizer

    all_results = {}

    for task_name in tasks:
        builder = TASK_BUILDERS[task_name]
        all_results[task_name] = {}

        for ctx in contexts:
            all_results[task_name][ctx] = {}

            for cond_name in conditions:
                env_vars = CONDITIONS.get(cond_name, {})
                # Apply condition env vars
                for k, v in env_vars.items():
                    os.environ[k] = v

                rng = random.Random(seed)
                scores_list = []
                times_list = []

                print(f"  [{task_name:20s}] ctx={ctx//1024}k cond={cond_name}",
                      end="", flush=True)

                for sample_i in range(num_samples):
                    example = builder(tokenizer, ctx, rng)
                    sid = f"cad_session_{sample_i}"
                    wrapper.clear_session(sid)
                    wrapper.active_session = sid

                    t0 = time.perf_counter()
                    response = wrapper.generate(
                        prompt=example["prompt"],
                        max_new_tokens=32,
                        temperature=0.0,
                        top_p=1.0,
                        repetition_penalty=1.0,
                    )
                    elapsed = time.perf_counter() - t0
                    times_list.append(elapsed)

                    s = score(response, example["answer"], example["check"])
                    scores_list.append(s)

                    wrapper.clear_session(sid)
                    mx.eval()
                    mx.clear_cache()
                    gc.collect()

                    if (sample_i + 1) % 5 == 0:
                        print(f" .{sample_i+1}", end="", flush=True)

                acc = sum(scores_list) / len(scores_list) * 100.0
                mean_t = sum(times_list) / len(times_list) if times_list else 0.0
                print(f"  → {acc:.1f}% ({mean_t:.1f}s/sample)", flush=True)

                all_results[task_name][ctx][cond_name] = {
                    "accuracy": round(acc, 1),
                    "mean_time_s": round(mean_t, 2),
                    "scores": scores_list,
                    "n_samples": num_samples,
                }

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("CAD Ablation Summary — accuracy (%) per condition", flush=True)
    print("=" * 80, flush=True)
    cond_cols = conditions
    header = f"{'Task':<22} {'Context':>8}" + "".join(
        f"  {c:>12}" for c in cond_cols
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for task_name in tasks:
        for ctx in contexts:
            row = f"{task_name:<22} {str(ctx//1024)+'k':>8}"
            for cond_name in cond_cols:
                acc = all_results[task_name].get(ctx, {}).get(
                    cond_name, {}).get("accuracy", float("nan"))
                row += f"  {acc:>12.1f}"
            print(row, flush=True)

    # Delta summary: full_cad vs baseline
    if "baseline" in conditions and "full_cad" in conditions:
        print("\n  Δ full_cad − baseline:", flush=True)
        total_delta = 0.0
        n_delta = 0
        for task_name in tasks:
            for ctx in contexts:
                b = all_results[task_name].get(ctx, {}).get("baseline", {}).get("accuracy")
                f = all_results[task_name].get(ctx, {}).get("full_cad", {}).get("accuracy")
                if b is not None and f is not None:
                    d = f - b
                    total_delta += d
                    n_delta += 1
                    print(f"    {task_name:20s} {ctx//1024}k:  {d:+.1f}pp", flush=True)
        if n_delta:
            print(f"    {'MEAN':20s}    :  {total_delta/n_delta:+.1f}pp", flush=True)

    # Save
    out = {
        "model": model_id,
        "contexts": contexts,
        "tasks": tasks,
        "conditions": conditions,
        "num_samples": num_samples,
        "seed": seed,
        "results": all_results,
        "condition_defs": {k: v for k, v in CONDITIONS.items() if k in conditions},
    }
    if output_path is None:
        output_path = os.path.join(REPO, "benchmarks", "results", "cad_ablation.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results → {output_path}", flush=True)
    return out


# ── Smoke test (no model inference) ───────────────────────────────────────────
def smoke_test():
    """Quick sanity check: build prompts for all tasks and score them."""
    print("CAD ablation smoke test (no model inference)", flush=True)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            trust_remote_code=True)
    except Exception:
        # Fake tokenizer for CI
        class FakeTok:
            def encode(self, t, add_special_tokens=False): return list(t.encode())
            def decode(self, ids): return bytes(ids).decode(errors="ignore")
        tok = FakeTok()

    rng = random.Random(0)
    for task_name, builder in TASK_BUILDERS.items():
        ex = builder(tok, 1000, rng)
        s = score(ex["answer"], ex["answer"], ex["check"])
        assert s == 1.0, f"Self-score failed for {task_name}"
        print(f"  {task_name:25s}  prompt_chars={len(ex['prompt']):6d}  self_score={s:.1f}  OK",
              flush=True)
    print("\nSmoke test PASSED", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Content-Aware Decision (CAD) residual selection ablation on MLX.")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--tasks", nargs="+", default=list(TASK_BUILDERS.keys()))
    ap.add_argument("--conditions", nargs="+", default=["baseline", "shape_only", "full_cad"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run prompt-building sanity check only (no model inference)")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        run_cad_ablation(
            model_id=args.model,
            contexts=args.contexts,
            num_samples=args.num_samples,
            tasks=args.tasks,
            conditions=args.conditions,
            output_path=args.output,
            seed=args.seed,
        )
