#!/usr/bin/env python3
"""Table-row binding probe v2 — NATIVE (C++) runner.

Reuses table_probe2's prompt builders and scorers; drives the native binary
(diffkv_native <model.gguf> <prompt>) exactly like native_margin_probe.sh:
temperature 0, DIFFKV_NATIVE_ATTN=0, one isolated process per cell.

Modes:
  dense  — DIFFKV_DENSE_CMP=1 (cpu dense reference path)
  diffkv — default compressed path
Table capture A/B via --extra-env DIFFKV_RESIDUAL_TABLE_CAPTURE=1.

Usage:
  python3 benchmarks/table_probe_native.py --ctx 16384 --modes dense diffkv
  python3 benchmarks/table_probe_native.py --ctx 16384 --modes diffkv \
      --extra-env DIFFKV_RESIDUAL_TABLE_CAPTURE=1
"""
import os, sys, json, argparse, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import table_probe2 as tp

BINARY = os.path.join(REPO, "diffkv_native", "build", "diffkv_native")
MODEL = os.path.join(REPO, "diffkv_native", "qwen2.5-1.5b-instruct-q8_0.gguf")

BASE_ENV = {
    "DIFFKV_ENGAGE_THRESHOLD": "1024",
    "DIFFKV_NATIVE_ATTN": "0",
    "DIFFKV_FORCE_CPU_ATTN": "0",
    "DIFFKV_MPS_APPROXIMATE_ATTN": "1",
    "DIFFKV_DENSE_DIRECT": "1",
    "DIFFKV_POOL_ABS_ROT": "1",
    "DIFFKV_TEMPERATURE": "0",
    "DIFFKV_DISABLE_VSL": "1",
    "DIFFKV_ENABLE_FACTUAL": "0",
    "DIFFKV_REPETITION_PENALTY": "1.0",
    "HF_HUB_OFFLINE": "1",
}

def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

def run_cell(tok, mode, ctx, question, max_tokens, extra_env, timeout=1800):
    prompt = tp.build_prompt(tok, ctx, question)
    # binary unescapes \\ then \n (see main.cpp ~2469) — escape in that order
    prompt = prompt.replace("\\", "\\\\").replace("\n", "\\n")
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(extra_env)   # extra-env overrides BASE_ENV (e.g. penalty A/Bs)
    env["DIFFKV_MAX_TOKENS"] = str(max_tokens)
    if mode == "dense":
        env["DIFFKV_DENSE_CMP"] = "1"
    proc = subprocess.Popen([BINARY, MODEL, prompt], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()   # pre-existing intermittent hang-at-exit: reap and salvage
        out, err = proc.communicate()
    text = out.decode("utf-8", errors="replace")
    # binary echoes the prompt then the generation; strip the prompt prefix
    tail = text
    for marker in ("assistant\n", "assistant:", "<|im_start|>assistant"):
        idx = text.rfind(marker)
        if idx >= 0:
            tail = text[idx + len(marker):]
            break
    return tail.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--modes", nargs="+", default=["dense", "diffkv"])
    ap.add_argument("--extra-env", nargs="*", default=[])
    args = ap.parse_args()
    extra_env = dict(kv.split("=", 1) for kv in args.extra_env)

    tok = get_tokenizer()
    summary = {}
    for mode in args.modes:
        t0 = time.time()
        print(f"\n=== NATIVE MODE {mode} ctx={args.ctx} extra={extra_env} ===", flush=True)
        ans = run_cell(tok, mode, args.ctx, tp.LIST_A_Q, 280, extra_env)
        table, fab_rows = tp.score_list_a(ans)
        ca = {k: sum(1 for v in table.values() if v == k)
              for k in ("correct", "partial", "mixed", "swap", "fab", "miss")}
        print(f"LIST-A [native/{mode}]: {ca}  fabricated_rows={fab_rows}")
        print(f"  per-row: {table}")
        print(f"  output: {ans[:500]!r}", flush=True)
        ans_b = run_cell(tok, mode, args.ctx, tp.LIST_B_Q, 160, extra_env)
        table_b, order_ok = tp.score_list_b(ans_b)
        cb = {k: sum(1 for v in table_b.values() if v == k)
              for k in ("correct", "swap", "fab", "miss")}
        print(f"LIST-B [native/{mode}]: {cb}  order_ok={order_ok}")
        print(f"  output: {ans_b[:400]!r}", flush=True)
        a_seq = run_cell(tok, mode, args.ctx, tp.SEQ_B_Q, 16, extra_env)
        v_seq = tp.classify_num(a_seq, "79.4", tp.ALL_VALS - {"79.4"})
        print(f"SEQ-B [native/{mode}]: {v_seq}  ({a_seq.strip()[:40]!r})", flush=True)
        summary[mode] = {"listA": ca, "fab_rows": fab_rows, "listB": cb,
                         "order_ok": order_ok, "seqB": v_seq}
        print(f"  mode wall time {time.time()-t0:.0f}s", flush=True)

    print("\n===== NATIVE TABLE PROBE SUMMARY =====")
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
