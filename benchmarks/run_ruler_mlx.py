#!/usr/bin/env python3
"""
run_ruler_mlx.py — RULER benchmark for DKV ACTIVE_RUNTIME (MLX)
====================================================================
RULER (Realistic, Unified, Long-context Evaluation and Retrieval) tests 8 task
types that go well beyond simple NIAH:

  Retrieval tasks (needle-style):
    niah_single_1   — single needle, one haystack document
    niah_single_2   — single needle, multi-document haystack
    niah_single_3   — single needle, needle is a paragraph (not a passcode)
    niah_multikey   — 2 needles, retrieve the one matching a query key
    niah_multivalue — 1 needle key → multiple possible values, return the right one
    niah_multiquery — 4 query keys, all present; return all matching values

  Aggregation tasks (need to process the whole context):
    variable_tracking — LISP-style variable assignment chain; find final value
    common_word_extraction (CWE) — count word frequencies, return top-k
    frequent_word_extraction (FWE) — same family, different k
    qa — single-hop QA over a long document

We implement a clean, self-contained subset:
  niah_single   — passcode buried in haystack (analogous to paper's E2)
  niah_multikey — 2 needles; model must use a query to pick the right one
  niah_multivalue — same key, two plausible values; model must get the right one
  variable_tracking — variable chain, find terminal value
  cwe           — count + return the most common word in a long list

Reference: Hsieh et al. (2024) "RULER: What's the Real Context Window Size
of Your LLM?" https://arxiv.org/abs/2404.06654

Runs DKV (COMPRESSED_DECODE=1) vs Dense (COMPRESSED_DECODE=0) side-by-side
on the same examples. Saves full per-example details plus aggregate scores.
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
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Filler corpus ─────────────────────────────────────────────────────────────
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def rand_code(rng, length=12):
    """Random alphanumeric code like ALPHA-1234-BETA."""
    p1 = ''.join(rng.choices(string.ascii_uppercase, k=5))
    p2 = ''.join(rng.choices(string.digits, k=4))
    p3 = ''.join(rng.choices(string.ascii_uppercase, k=4))
    return f"{p1}-{p2}-{p3}"

def pad_to_tokens(tokenizer, text_parts: list, target_tokens: int, rng: random.Random) -> str:
    """Interleave text_parts with filler to reach ~target_tokens total."""
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    # measure current length
    parts_toks = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in text_parts)
    budget = max(50, target_tokens - parts_toks - 80)
    reps = (budget // len(filler_toks)) + 2
    all_filler = (filler_toks * reps)[:budget]
    filler_text = tokenizer.decode(all_filler)
    # splice parts evenly through filler
    n = len(text_parts)
    chunk = len(filler_text) // (n + 1)
    out = []
    for i, part in enumerate(text_parts):
        out.append(filler_text[i * chunk:(i + 1) * chunk])
        out.append(part)
    out.append(filler_text[n * chunk:])
    return "".join(out)

def qwen_prompt(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ── Task builders ─────────────────────────────────────────────────────────────

def build_niah_single(tokenizer, target_tokens: int, rng: random.Random) -> dict:
    """Single needle: one passcode buried in filler. Answer = passcode."""
    code = rand_code(rng)
    needle_sent = f"The secret passcode is {code}."
    question = "What is the secret passcode? Repeat it exactly, nothing else."
    needle_part = "\n" + needle_sent + "\n"
    body = pad_to_tokens(tokenizer, [needle_part], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "niah_single", "answer": code, "prompt": prompt, "check": "exact_in"}

def build_niah_multikey(tokenizer, target_tokens: int, rng: random.Random) -> dict:
    """Two needles with different keys; question specifies which key to retrieve."""
    key_a, code_a = rand_code(rng), rand_code(rng)
    key_b, code_b = rand_code(rng), rand_code(rng)
    needle_a = f"\nThe passcode for key {key_a} is {code_a}.\n"
    needle_b = f"\nThe passcode for key {key_b} is {code_b}.\n"
    question = f"What is the passcode for key {key_b}? State only the passcode."
    body = pad_to_tokens(tokenizer, [needle_a, needle_b], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "niah_multikey", "answer": code_b, "prompt": prompt, "check": "exact_in"}

def build_niah_multivalue(tokenizer, target_tokens: int, rng: random.Random) -> dict:
    """One key with a value buried early; distractor value buried later. Answer = first."""
    key = rand_code(rng)
    val_correct = rand_code(rng)
    val_distract = rand_code(rng)
    needle_correct  = f"\nThe authoritative passcode for {key} is {val_correct}.\n"
    needle_distract = f"\nAn outdated entry suggests the passcode for {key} might be {val_distract}.\n"
    question = f"What is the authoritative passcode for {key}? State only the passcode."
    body = pad_to_tokens(tokenizer, [needle_correct, needle_distract], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "niah_multivalue", "answer": val_correct, "prompt": prompt, "check": "exact_in"}

def build_variable_tracking(tokenizer, target_tokens: int, rng: random.Random) -> dict:
    """
    Variable assignment chain: x0=A, x1=x0, x2=x1, ..., xN=xN-1.
    Answer is the terminal value (the initial assignment).
    """
    depth = 8
    final_val = rand_code(rng)
    # build chain sentences
    chain = [f"Let x0 = {final_val}."]
    for i in range(1, depth):
        chain.append(f"Let x{i} = x{i - 1}.")
    chain_text = " ".join(chain)
    question = f"What is the value of x{depth - 1}? State only the value."
    chain_part = "\n" + chain_text + "\n"
    body = pad_to_tokens(tokenizer, [chain_part], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "variable_tracking", "answer": final_val, "prompt": prompt, "check": "exact_in"}

def build_cwe(tokenizer, target_tokens: int, rng: random.Random) -> dict:
    """
    Common Word Extraction: inject a target word ~20x and many distractors ~3x.
    Ask for the most frequently occurring word.
    """
    target_word = rng.choice(["ZEPHYR", "AURORA", "QUANTUM", "NEBULA", "VORTEX"])
    other_words = [w for w in ["ARCTIC", "BEACON", "CRYSTAL", "DELTA", "EMBER",
                                "FLARE", "GRAVEL", "HELIX", "INDIGO", "JADE"] if w != target_word]
    # build a word-list passage
    word_list = ([target_word] * 20 + other_words * 3)
    rng.shuffle(word_list)
    passage = " ".join(word_list)
    passage_part = f"\nWord list: {passage}\n"
    question = (
        "In the word list above, which single word appears most frequently? "
        "State only that word, nothing else."
    )
    body = pad_to_tokens(tokenizer, [passage_part], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + question)
    return {"task": "cwe", "answer": target_word, "prompt": prompt, "check": "exact_in_ci"}

TASK_BUILDERS = {
    "niah_single":      build_niah_single,
    "niah_multikey":    build_niah_multikey,
    "niah_multivalue":  build_niah_multivalue,
    "variable_tracking": build_variable_tracking,
    "cwe":              build_cwe,
}

# ── Scoring ────────────────────────────────────────────────────────────────────
def score(prediction: str, answer: str, check: str) -> float:
    pred = prediction.strip()
    if check == "exact_in":
        return 1.0 if answer in pred else 0.0
    if check == "exact_in_ci":
        return 1.0 if answer.lower() in pred.lower() else 0.0
    return 0.0

# ── Runner ────────────────────────────────────────────────────────────────────
def run_ruler(
    model_id: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    contexts: list = None,
    num_samples: int = 10,
    tasks: list = None,
    output_path: str = None,
    seed: int = 42,
):
    if contexts is None:
        contexts = [4096, 8192, 16384]
    if tasks is None:
        tasks = list(TASK_BUILDERS.keys())

    print(f"RULER — DKV vs Dense", flush=True)
    print(f"Model:    {model_id}", flush=True)
    print(f"Contexts: {contexts}", flush=True)
    print(f"Tasks:    {tasks}", flush=True)
    print(f"Samples:  {num_samples} per (task × context × mode)\n", flush=True)

    from serving.mlx_dkv_wrapper import MLXDKVWrapper

    # Load wrapper once (shared model weights)
    wrapper = MLXDKVWrapper(model_id=model_id, config={"rank": 32, "block_size": 256})
    wrapper.ensure_loaded()
    tokenizer = wrapper.tokenizer

    all_results = {}
    modes = [
        ("dkv", {"DKV_COMPRESSED_DECODE": "1", "DKV_MAX_RESIDUAL": "128",
                    "DKV_SPARSE_PREFILL": "1", "DKV_DECODE_CACHE": "1",
                    "DKV_SPARSE_BIAS": "auto", "DKV_SEED": "1234"}),
        ("dense",  {"DKV_COMPRESSED_DECODE": "0"}),
    ]

    for task_name in tasks:
        builder = TASK_BUILDERS[task_name]
        all_results[task_name] = {}

        for ctx in contexts:
            all_results[task_name][ctx] = {}

            for mode_name, env_vars in modes:
                for k, v in env_vars.items():
                    os.environ[k] = v

                rng = random.Random(seed)
                scores = []
                times = []

                print(f"  [{task_name}] ctx={ctx//1024}k mode={mode_name}", end="", flush=True)

                for sample_i in range(num_samples):
                    example = builder(tokenizer, ctx, rng)
                    sid = "ruler_session"
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
                    times.append(elapsed)

                    s = score(response, example["answer"], example["check"])
                    scores.append(s)

                    wrapper.clear_session(sid)
                    # Flush Metal GPU cache + Python garbage collector between samples
                    import gc
                    mx.eval()
                    mx.clear_cache()
                    gc.collect()

                    if (sample_i + 1) % 5 == 0:
                        print(f" .{sample_i+1}", end="", flush=True)

                acc = sum(scores) / len(scores) * 100.0
                mean_t = sum(times) / len(times)
                print(f"  → {acc:.1f}% ({mean_t:.1f}s/sample)", flush=True)

                all_results[task_name][ctx][mode_name] = {
                    "accuracy": round(acc, 1),
                    "mean_time_s": round(mean_t, 2),
                    "scores": scores,
                }

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("RULER Summary — DKV vs Dense accuracy (%)", flush=True)
    print("=" * 70, flush=True)
    header = f"{'Task':<22}" + "".join(
        f"  {ctx//1024}k DKV  {ctx//1024}k Dense" for ctx in contexts
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for task_name in tasks:
        row = f"{task_name:<22}"
        for ctx in contexts:
            d = all_results[task_name].get(ctx, {})
            dk = d.get("dkv", {}).get("accuracy", float("nan"))
            dn = d.get("dense", {}).get("accuracy", float("nan"))
            row += f"  {dk:>8.1f}  {dn:>8.1f}"
        print(row, flush=True)

    # Save
    out = {
        "model": model_id,
        "contexts": contexts,
        "tasks": tasks,
        "num_samples": num_samples,
        "seed": seed,
        "results": all_results,
    }
    if output_path is None:
        output_path = os.path.join(REPO, "benchmarks", "results", "ruler_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved RULER results to {output_path}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384])
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--tasks", nargs="+", default=list(TASK_BUILDERS.keys()))
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_ruler(
        model_id=args.model,
        contexts=args.contexts,
        num_samples=args.num_samples,
        tasks=args.tasks,
        output_path=args.output,
        seed=args.seed,
    )
