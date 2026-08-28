#!/usr/bin/env python3
"""
run_int4_residual_matrix.py — Production-Grade Controlled Benchmark Matrix
==========================================================================
Compares 3 residual configurations across 4K, 8K, 16K, 32K context lengths:

  1. FP16 (R=32)    — Standard baseline at budget R=32 (FP16 storage)
  2. INT4 (R=32)    — 4-bit physically packed residuals at R=32 (3.56x less residual memory)
  3. INT4 (R=128)   — 4-bit physically packed residuals at R=128 (4x slots in ~same memory)

Tasks:
  - exact_numeric   — 6-digit numeric passkey in structured JSON/metrics filler (ruler_kv)
  - multi_key       — two distinct keys with distinct passcodes (multi_niah)
  - multi_value     — single key with authoritative value + distractor value (niah_multivalue)
  - variable_track  — 8-step assignment chain x0=A, x1=x0, ..., x7=x6 (variable_tracking)

Metrics:
  - Exact Retrieval Accuracy (%)
  - Physical Residual Pool Memory (MB) — exact .nbytes of allocated residual buffers
  - Total Active Metal Resident Memory (MB) via mx.get_active_memory()
  - Latency (s)
"""

import os
import sys
import gc
import json
import time
import random
import string
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import mlx.core as mx
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
except ImportError as e:
    print(f"ERROR: MLX import failed: {e}")
    sys.exit(1)

def get_metal_mem_mb():
    try:
        mem_fn = getattr(mx, "get_active_memory", None) or getattr(mx.metal, "get_active_memory", None)
        if mem_fn is not None:
            return mem_fn() / (1024 * 1024)
    except Exception:
        pass
    return 0.0

def flush_cache():
    try:
        clear_fn = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        if clear_fn is not None:
            clear_fn()
    except Exception:
        pass
    gc.collect()

def get_session_res_mb(session, manager):
    if not session:
        return 0.0
    total_bytes = 0
    if session.get("comp_res_k") is not None and session["comp_res_k"][0] is not None:
        for l in range(manager.num_layers):
            total_bytes += session["comp_res_k"][l].nbytes + session["comp_res_v"][l].nbytes
    elif session.get("comp_res_k_q") is not None and session["comp_res_k_q"][0] is not None:
        for l in range(manager.num_layers):
            total_bytes += session["comp_res_k_q"][l].nbytes + session["comp_res_k_s"][l].nbytes + session["comp_res_k_b"][l].nbytes
            total_bytes += session["comp_res_v_q"][l].nbytes + session["comp_res_v_s"][l].nbytes + session["comp_res_v_b"][l].nbytes
    return total_bytes / (1024 * 1024)

# ── Corpus Filler ──────────────────────────────────────────────────────────────
FILLER = (
    "The analysis of high-dimensional empirical risk minimization relies on "
    "spectral decay properties of the covariance operator. "
    "In reproducing kernel Hilbert spaces, regularization parameters balance "
    "approximation bias against statistical estimation variance. "
    "Random feature expansions construct finite-dimensional approximations of continuous kernels. "
    "Empirical spectral distributions exhibit universality across non-Gaussian ensembles. "
    "Gradient flow dynamics on non-convex energy landscapes converge to stationary points. "
    "Information-theoretic bounds establish minimax optimal rates for distributed estimation. "
)

def rand_code(rng, length=12):
    p1 = "".join(rng.choices(string.ascii_uppercase, k=5))
    p2 = "".join(rng.choices(string.digits, k=4))
    p3 = "".join(rng.choices(string.ascii_uppercase, k=4))
    return f"{p1}-{p2}-{p3}"

def pad_to_tokens(tokenizer, text_parts, target_tokens, rng):
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    parts_toks = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in text_parts)
    budget = max(50, target_tokens - parts_toks - 60)
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

def build_exact_numeric(tokenizer, target_tokens, rng):
    code = "".join(rng.choices(string.digits, k=6))
    json_filler = '{"id": "rec_01", "metrics": [12.4, 99.1, 0.05], "status": "active"}\n'
    needle = f'\n"authorization_code": "{code}"\n'
    q = 'What is the value of "authorization_code" in the data above? State only the 6-digit number.'
    body = pad_to_tokens(tokenizer, [json_filler * 4 + needle], target_tokens, rng)
    prompt = qwen_prompt("You are a precise data extraction system.", body + "\n\n" + q)
    return {"task": "exact_numeric", "answer": code, "prompt": prompt}

def build_multi_key(tokenizer, target_tokens, rng):
    key_a, val_a = rand_code(rng), rand_code(rng)
    key_b, val_b = rand_code(rng), rand_code(rng)
    needle_a = f"\nThe security passcode for key {key_a} is {val_a}.\n"
    needle_b = f"\nThe security passcode for key {key_b} is {val_b}.\n"
    q = f"What is the security passcode for key {key_b}? State only the passcode."
    body = pad_to_tokens(tokenizer, [needle_a, needle_b], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + q)
    return {"task": "multi_key", "answer": val_b, "prompt": prompt}

def build_multi_value(tokenizer, target_tokens, rng):
    key = rand_code(rng)
    val_correct = rand_code(rng)
    val_distract = rand_code(rng)
    needle_correct = f"\nThe authoritative passcode for {key} is {val_correct}.\n"
    needle_distract = f"\nAn outdated log entry suggests the passcode for {key} might be {val_distract}.\n"
    q = f"What is the authoritative passcode for {key}? State only the passcode."
    body = pad_to_tokens(tokenizer, [needle_correct, needle_distract], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + q)
    return {"task": "multi_value", "answer": val_correct, "prompt": prompt}

def build_variable_tracking(tokenizer, target_tokens, rng):
    depth = 8
    final_val = rand_code(rng)
    chain = [f"Let x0 = {final_val}."]
    for i in range(1, depth):
        chain.append(f"Let x{i} = x{i - 1}.")
    chain_text = " ".join(chain)
    q = f"What is the value of x{depth - 1}? State only the value."
    body = pad_to_tokens(tokenizer, ["\n" + chain_text + "\n"], target_tokens, rng)
    prompt = qwen_prompt("You are a helpful assistant.", body + "\n\n" + q)
    return {"task": "variable_track", "answer": final_val, "prompt": prompt}

TASK_BUILDERS = {
    "exact_numeric":  build_exact_numeric,
    "multi_key":      build_multi_key,
    "multi_value":    build_multi_value,
    "variable_track": build_variable_tracking,
}

CONFIGURATIONS = {
    "FP16 (R=32)": {
        "config": {"rank": 32, "block_size": 1024, "max_residual": 32, "residual_quant": "none"},
        "env": {"DKV_COMPRESSED_DECODE": "1", "DKV_DECODE_CACHE": "1", "DKV_SPARSE_PREFILL": "1"}
    },
    "INT4 (R=32)": {
        "config": {"rank": 32, "block_size": 1024, "max_residual": 32, "residual_quant": "int4"},
        "env": {"DKV_COMPRESSED_DECODE": "1", "DKV_DECODE_CACHE": "1", "DKV_SPARSE_PREFILL": "1"}
    },
    "INT4 (R=128)": {
        "config": {"rank": 32, "block_size": 1024, "max_residual": 128, "residual_quant": "int4"},
        "env": {"DKV_COMPRESSED_DECODE": "1", "DKV_DECODE_CACHE": "1", "DKV_SPARSE_PREFILL": "1"}
    },
}

def run_matrix(
    model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    contexts=None,
    tasks=None,
    num_samples=2,
    seed=42,
):
    if contexts is None:
        contexts = [4096, 8192, 16384]
    if tasks is None:
        tasks = ["exact_numeric", "multi_key", "multi_value", "variable_track"]

    print("=" * 85)
    print("4-Bit Quantized Residual Matrix Benchmark — Physical Storage & Accuracy Matrix")
    print("=" * 85)
    print(f"Model:        {model_id}")
    print(f"Contexts:     {[str(c//1024)+'k' for c in contexts]}")
    print(f"Tasks:        {tasks}")
    print(f"Config Arms:  {list(CONFIGURATIONS.keys())}")
    print(f"Samples:      {num_samples} per (task × context × arm)\n")

    # Load wrappers for each configuration arm
    wrappers = {}
    for arm_name, spec in CONFIGURATIONS.items():
        print(f"Loading wrapper for {arm_name} (max_residual={spec['config']['max_residual']}, quant={spec['config']['residual_quant']})...")
        for k, v in spec["env"].items():
            os.environ[k] = v
        w = MLXDKVWrapper(model_id=model_id, config=spec["config"])
        w.ensure_loaded()
        wrappers[arm_name] = w

    tok = list(wrappers.values())[0].tokenizer
    results = {}

    for task_name in tasks:
        builder = TASK_BUILDERS[task_name]
        results[task_name] = {}
        print(f"\n▶ Task: {task_name}")

        for ctx in contexts:
            results[task_name][ctx] = {}
            ctx_str = f"{ctx//1024}k"

            for arm_name, spec in CONFIGURATIONS.items():
                wrapper = wrappers[arm_name]
                for k, v in spec["env"].items():
                    os.environ[k] = v

                rng = random.Random(seed)
                scores = []
                times = []
                memories = []
                res_pools = []

                print(f"  [{task_name:14s}] {ctx_str:>3s} | {arm_name:14s}", end="", flush=True)

                for sample_i in range(num_samples):
                    ex = builder(tok, ctx, rng)
                    sid = f"matrix_sess_{task_name}_{ctx}_{sample_i}"
                    wrapper.clear_session(sid)
                    wrapper.active_session = sid

                    flush_cache()
                    t0 = time.perf_counter()
                    resp = wrapper.generate(
                        prompt=ex["prompt"],
                        max_new_tokens=32,
                        temperature=0.0,
                    )
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)

                    mem_active = get_metal_mem_mb()
                    memories.append(mem_active)

                    sess = wrapper.manager.sessions.get(sid, {})
                    res_mb = get_session_res_mb(sess, wrapper.manager)
                    res_pools.append(res_mb)

                    ans = resp.strip()
                    if "assistant\n" in ans:
                        ans = ans.split("assistant\n")[-1].strip()

                    correct = ex["answer"] in ans
                    scores.append(1.0 if correct else 0.0)

                    wrapper.clear_session(sid)
                    flush_cache()
                    print(f" .{sample_i+1}", end="", flush=True)

                acc = (sum(scores) / len(scores)) * 100.0
                mean_t = sum(times) / len(times)
                mean_mem = sum(memories) / len(memories) if memories else 0.0
                mean_res = sum(res_pools) / len(res_pools) if res_pools else 0.0

                print(f"  → Acc: {acc:5.1f}% | ResPool: {mean_res:5.2f}MB | MetalMem: {mean_mem:6.1f}MB | {mean_t:4.1f}s", flush=True)

                results[task_name][ctx][arm_name] = {
                    "accuracy": acc,
                    "res_pool_mb": mean_res,
                    "metal_mem_mb": mean_mem,
                    "mean_time_s": mean_t,
                    "scores": scores,
                }

    # ── Summary Report Table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FINAL SUMMARY RESULTS MATRIX: Accuracy (%) | ResPool RAM (MB) | Total Metal RAM (MB)")
    print("=" * 100)
    header = f"{'Task':<16} {'Ctx':>4} | {'FP16 (R=32)':^24} | {'INT4 (R=32)':^24} | {'INT4 (R=128)':^24}"
    print(header)
    print("-" * len(header))
    for task_name in tasks:
        for ctx in contexts:
            ctx_str = f"{ctx//1024}k"
            row = f"{task_name:<16} {ctx_str:>4} |"
            for arm in ["FP16 (R=32)", "INT4 (R=32)", "INT4 (R=128)"]:
                d = results[task_name].get(ctx, {}).get(arm, {})
                acc = d.get("accuracy", float("nan"))
                res_mb = d.get("res_pool_mb", 0.0)
                row += f" {acc:>5.1f}% (Res:{res_mb:4.1f}M) |"
            print(row)

    out_path = os.path.join(REPO, "benchmarks", "results", "int4_residual_matrix_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved matrix results to {out_path}")
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384])
    ap.add_argument("--tasks", nargs="+", default=["exact_numeric", "multi_key", "multi_value", "variable_track"])
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_matrix(
        model_id=args.model,
        contexts=args.contexts,
        tasks=args.tasks,
        num_samples=args.num_samples,
        seed=args.seed,
    )
