#!/usr/bin/env python3
"""
clean_sweep_v2.py — Symmetric single-process benchmark for DiffKV paper.

Both DiffKV (active) and Dense use their respective generate() function —
exactly how each engine is invoked from the CLI. Timing methodology is
identical for both:

  1. Warmup: generate(1 token) on a short ~64-token prompt to compile Metal
             kernels. This runs once per engine-context cell.
  2. TTFT pass: generate(1 token) on the full-length benchmark prompt.
               wall-clock = prefill + 1 decode step  ≈ prefill_s.
  3. Full pass: generate(GEN tokens) on the same full-length prompt (fresh
               session/cache — independent of TTFT pass).
               wall-clock = prefill + GEN decode steps = total_s.
  4. Derived:
       decode_per_step = (total_s - ttft_s) / (GEN - 1)
       prefill_s        = ttft_s - decode_per_step
       decode_tps       = GEN / (decode_per_step * GEN)

  Between every cell: full session/cache teardown + gc.collect + mx.clear_cache
  to guarantee zero bleed-over between runs.

Run order: DiffKV 4k→64k, then Dense 4k→64k.
Each context runs in an isolated subprocess for maximum memory isolation.

Usage:
  python3 benchmarks/clean_sweep_v2.py [--contexts 4096 ...] [--gen 128]
  python3 benchmarks/clean_sweep_v2.py --engines active   # DiffKV only
  python3 benchmarks/clean_sweep_v2.py --engines dense    # Dense only
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
VENV_PY = os.path.join(REPO, "diffkv_venv", "bin", "python3")
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CONTEXTS = [4096, 8192, 16384, 32768, 65536]
GEN = 128
NEEDLE = "OMEGA-7741-DELTA"


# ── Active (DiffKV) worker ────────────────────────────────────────────────────
ACTIVE_WORKER = r'''
import sys, os, json, time, gc
REPO, CTX, GEN, MODEL_ID, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
NEEDLE = "OMEGA-7741-DELTA"

sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import mlx.core as mx

os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
os.environ["DIFFKV_MAX_RESIDUAL"] = "128"
os.environ["DIFFKV_SPARSE_PREFILL"] = "1"
os.environ["DIFFKV_DECODE_CACHE"] = "1"
os.environ["DIFFKV_SPARSE_BIAS"] = "auto"
os.environ["DIFFKV_SEED"] = "1234"

from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
wrapper = MLXDiffKVWrapper(model_id=MODEL_ID, config={"rank": 32, "block_size": 256})
wrapper.ensure_loaded()

# Load prompt from saved file (written by earlier sweep)
pfile = os.path.join(REPO, "benchmarks", "results", f"prompt_{CTX}.txt")
with open(pfile) as f:
    prompt_text = f.read()
prompt_tokens = len(wrapper.tokenizer.encode(prompt_text))
print(f"[active {CTX//1024}k] prompt_tokens={prompt_tokens}", flush=True)

def mx_peak():
    for obj, nm in ((mx, "get_peak_memory"), (getattr(mx,"metal",None), "get_peak_memory")):
        if obj and hasattr(obj, nm):
            try: return float(getattr(obj,nm)()) / 1e9
            except: pass
    return None

def mx_reset():
    for obj, nm in ((mx, "reset_peak_memory"), (getattr(mx,"metal",None), "reset_peak_memory")):
        if obj and hasattr(obj, nm):
            try: getattr(obj,nm)()
            except: pass

def clear_all(sid):
    wrapper.clear_session(sid)
    # Also purge the token-ID cache the wrapper uses for prefill reuse
    # (wrapper.generate checks _session_token_ids to skip re-prefill)
    if hasattr(wrapper, '_session_token_ids'):
        wrapper._session_token_ids.pop(sid, None)
    gc.collect()
    mx.clear_cache()
    mx.eval()

# ── Step 1: Warmup — tiny prompt, 1 token, compiles Metal kernels ─────────
print(f"[active {CTX//1024}k] warmup...", flush=True)
sid_w = "bench_warm"
clear_all(sid_w)
wrapper.active_session = sid_w
_ = wrapper.generate(prompt="Hello.", max_new_tokens=1, temperature=0.0)
clear_all(sid_w)
print(f"[active {CTX//1024}k] warmup done", flush=True)

# ── Step 2: TTFT pass — full prompt, 1 token ──────────────────────────────
# Use a FRESH session ID (never seen before) to guarantee no prefill reuse
mx_reset()
sid_a = f"bench_ttft_{CTX}"
clear_all(sid_a)
wrapper.active_session = sid_a
t0 = time.perf_counter()
_ = wrapper.generate(prompt=prompt_text, max_new_tokens=1, temperature=0.0)
mx.eval(mx.zeros(1))
ttft_s = time.perf_counter() - t0
clear_all(sid_a)
print(f"[active {CTX//1024}k] ttft={ttft_s:.3f}s", flush=True)

# ── Step 3: Full pass — fresh session, full prompt, GEN tokens ───────────
# Use another distinct session ID so wrapper cannot reuse TTFT's prefill
mx_reset()
sid_b = f"bench_full_{CTX}"
clear_all(sid_b)
wrapper.active_session = sid_b
t0 = time.perf_counter()
response = wrapper.generate(prompt=prompt_text, max_new_tokens=GEN, temperature=0.0)
mx.eval(mx.zeros(1))
total_s = time.perf_counter() - t0
peak = mx_peak()
clear_all(sid_b)
print(f"[active {CTX//1024}k] total={total_s:.3f}s peak={peak:.2f}GB", flush=True)

# ── Derived timing ─────────────────────────────────────────────────────────
n_extra = max(1, GEN - 1)
decode_per_step = max(0.001, (total_s - ttft_s) / n_extra)
prefill_s = max(0.0, ttft_s - decode_per_step)
decode_s = decode_per_step * GEN

result = {
    "prompt_tokens": prompt_tokens, "gen_tokens": GEN,
    "prefill_s": prefill_s, "decode_s": decode_s,
    "decode_tps": GEN / decode_s,
    "ttft_s": ttft_s, "mx_peak_gb": peak,
    "output_preview": response[:300],
    "needle_found": NEEDLE in response,
    "engine": "active", "ctx_target": CTX, "gen_target": GEN, "status": "ok",
    "_timing": f"ttft={ttft_s:.3f} total={total_s:.3f} dps={decode_per_step:.4f}",
}
with open(OUT, "w") as f: json.dump(result, f, indent=2)
print(f"[active {CTX//1024}k] prefill={prefill_s:.2f}s decode_tps={result['decode_tps']:.2f} needle={'PASS' if result['needle_found'] else 'FAIL'}", flush=True)
'''


# ── Dense (mlx_lm) worker — same generate() pattern ─────────────────────────
DENSE_WORKER = r'''
import sys, os, json, time, gc
REPO, CTX, GEN, MODEL_ID, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
NEEDLE = "OMEGA-7741-DELTA"

sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm import generate as mlx_generate

model, tokenizer = load(MODEL_ID)
prompt_tokens_n = None

pfile = os.path.join(REPO, "benchmarks", "results", f"prompt_{CTX}.txt")
with open(pfile) as f:
    prompt_text = f.read()
prompt_tokens = len(tokenizer.encode(prompt_text))
print(f"[dense {CTX//1024}k] prompt_tokens={prompt_tokens}", flush=True)

def mx_peak():
    for obj, nm in ((mx, "get_peak_memory"), (getattr(mx,"metal",None), "get_peak_memory")):
        if obj and hasattr(obj, nm):
            try: return float(getattr(obj,nm)()) / 1e9
            except: pass
    return None

def mx_reset():
    for obj, nm in ((mx, "reset_peak_memory"), (getattr(mx,"metal",None), "reset_peak_memory")):
        if obj and hasattr(obj, nm):
            try: getattr(obj,nm)()
            except: pass

def clear_all():
    gc.collect(); mx.clear_cache(); mx.eval()

# ── Step 1: Warmup — tiny prompt, 1 token ────────────────────────────────
print(f"[dense {CTX//1024}k] warmup...", flush=True)
_ = mlx_generate(model, tokenizer, prompt="Hello.", max_tokens=1, verbose=False, temp=0.0)
clear_all()
print(f"[dense {CTX//1024}k] warmup done", flush=True)

# ── Step 2: TTFT pass — full prompt, 1 token ─────────────────────────────
mx_reset()
t0 = time.perf_counter()
_ = mlx_generate(model, tokenizer, prompt=prompt_text, max_tokens=1, verbose=False, temp=0.0)
mx.eval(mx.zeros(1))
ttft_s = time.perf_counter() - t0
clear_all()
print(f"[dense {CTX//1024}k] ttft={ttft_s:.3f}s", flush=True)

# ── Step 3: Full pass — full prompt, GEN tokens ───────────────────────────
mx_reset()
t0 = time.perf_counter()
response = mlx_generate(model, tokenizer, prompt=prompt_text, max_tokens=GEN, verbose=False, temp=0.0)
mx.eval(mx.zeros(1))
total_s = time.perf_counter() - t0
peak = mx_peak()
clear_all()
print(f"[dense {CTX//1024}k] total={total_s:.3f}s peak={peak:.2f}GB", flush=True)

# ── Derived timing (identical formula to active) ───────────────────────────
n_extra = max(1, GEN - 1)
decode_per_step = max(0.001, (total_s - ttft_s) / n_extra)
prefill_s = max(0.0, ttft_s - decode_per_step)
decode_s = decode_per_step * GEN

result = {
    "prompt_tokens": prompt_tokens, "gen_tokens": GEN,
    "prefill_s": prefill_s, "decode_s": decode_s,
    "decode_tps": GEN / decode_s,
    "ttft_s": ttft_s, "mx_peak_gb": peak,
    "output_preview": response[:300],
    "needle_found": NEEDLE in response,
    "engine": "dense", "ctx_target": CTX, "gen_target": GEN, "status": "ok",
    "_timing": f"ttft={ttft_s:.3f} total={total_s:.3f} dps={decode_per_step:.4f}",
}
with open(OUT, "w") as f: json.dump(result, f, indent=2)
print(f"[dense {CTX//1024}k] prefill={prefill_s:.2f}s decode_tps={result['decode_tps']:.2f} needle={'PASS' if result['needle_found'] else 'FAIL'}", flush=True)
'''


def run_cell(engine, ctx, gen=GEN):
    out = os.path.join(RESULTS, f"clean_{engine}_{ctx}.json")
    code = ACTIVE_WORKER if engine == "active" else DENSE_WORKER
    print(f"\n{'='*60}", flush=True)
    print(f"  {engine.upper()} @ {ctx//1024}k", flush=True)
    print(f"{'='*60}", flush=True)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    proc = subprocess.run(
        [VENV_PY, "-c", code, REPO, str(ctx), str(gen), MODEL_ID, out],
        env=env, cwd=REPO,
    )
    if proc.returncode != 0:
        print(f"  [FAIL] exit {proc.returncode}", flush=True)
        return False
    if os.path.exists(out):
        with open(out) as f:
            r = json.load(f)
        if r.get("status") == "ok":
            print(f"  ✓  prefill={r['prefill_s']:.2f}s  "
                  f"decode={r['decode_tps']:.2f} tok/s  "
                  f"peak={r['mx_peak_gb']:.2f}GB  "
                  f"needle={'PASS' if r['needle_found'] else 'FAIL'}", flush=True)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    ap.add_argument("--gen", type=int, default=GEN)
    ap.add_argument("--engines", nargs="+", choices=["active", "dense"],
                    default=["active", "dense"])
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    summary = {}
    # Run all contexts for each engine before moving to the next engine
    for engine in args.engines:
        for ctx in args.contexts:
            ok = run_cell(engine, ctx, args.gen)
            summary[f"{engine}_{ctx}"] = "ok" if ok else "FAIL"

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE  ({elapsed:.0f}s / {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:25s}  {v}")


if __name__ == "__main__":
    main()
