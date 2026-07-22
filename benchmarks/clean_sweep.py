#!/usr/bin/env python3
"""
clean_sweep.py — Fresh paper benchmark. Direct single-pass timing, no hacks.

For active (DKV):
  - Use wrapper.generate() for a 1-token warmup (compiles Metal kernels)
  - Clear session, then time wrapper.generate(max_new_tokens=1) → prefill_s
  - Clear session, then run SEPARATE decode-only pass using wrapper.generate()
    but inferring decode_tps from (full_time - prefill_s) / gen_tokens

  Actually simpler: use wrapper.generate() with max_new_tokens=GEN, time it
  end-to-end. Then separately run with max_new_tokens=1 to get time-to-first-token.
  decode_s ≈ full_time - ttft_time
  This is clean and reproducible — same session cleared between every call.

For dense (mlx_lm):
  - Direct chunked prefill + token-by-token decode loop, each timed separately.

Writes benchmarks/results/clean_{engine}_{ctx}.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
VENV_PY = os.path.join(REPO, "dkv_venv", "bin", "python3")

CONTEXTS_DEFAULT = [4096, 8192, 16384, 32768, 65536]
GEN_DEFAULT = 128
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
NEEDLE_PASSCODE = "OMEGA-7741-DELTA"

# ── Inline worker (run in subprocess for isolation) ──────────────────────────

ACTIVE_WORKER = r'''
import sys, os, json, time, gc

REPO, CTX, GEN, MODEL_ID, OUT_PATH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
NEEDLE_PASSCODE = "OMEGA-7741-DELTA"

sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from bench_common import build_niah_prompt

def mx_peak():
    try:
        import mlx.core as mx
        for obj, name in ((mx, "get_peak_memory"), (getattr(mx,"metal",None), "get_peak_memory")):
            if obj and hasattr(obj, name):
                return float(getattr(obj, name)()) / 1e9
    except: pass
    return None

def mx_reset():
    try:
        import mlx.core as mx
        for obj, name in ((mx, "reset_peak_memory"), (getattr(mx,"metal",None), "reset_peak_memory")):
            if obj and hasattr(obj, name):
                try: getattr(obj, name)()
                except: pass
    except: pass

from serving.mlx_dkv_wrapper import MLXDKVWrapper
import mlx.core as mx

os.environ["DKV_COMPRESSED_DECODE"] = "1"
os.environ["DKV_MAX_RESIDUAL"] = "128"
os.environ["DKV_SPARSE_PREFILL"] = "1"
os.environ["DKV_DECODE_CACHE"] = "1"
os.environ["DKV_SPARSE_BIAS"] = "auto"
os.environ["DKV_SEED"] = "1234"

wrapper = MLXDKVWrapper(model_id=MODEL_ID, config={"rank": 32, "block_size": 256})
wrapper.ensure_loaded()

# Build prompt — build_niah_prompt returns (text, token_count)
try:
    from bench_common import build_niah_prompt
    _result = build_niah_prompt(CTX, wrapper.tokenizer)
    prompt_text = _result[0] if isinstance(_result, tuple) else _result
except Exception:
    pfile = os.path.join(REPO, "benchmarks", "results", f"prompt_{CTX}.txt")
    if os.path.exists(pfile):
        with open(pfile) as f:
            prompt_text = f.read()
    else:
        raise

prompt_tokens = len(wrapper.tokenizer.encode(prompt_text))
print(f"[active] prompt_tokens={prompt_tokens}", flush=True)

# ── Warmup: one throwaway call at tiny context to compile Metal kernels ──
_sid = "__warmup__"
wrapper.clear_session(_sid); wrapper.active_session = _sid
_ = wrapper.generate(prompt="Hello.", max_new_tokens=1, temperature=0.0)
wrapper.clear_session(_sid)
gc.collect(); mx.clear_cache()
print("[active] warmup done", flush=True)

# ── Pass A: time-to-first-token (TTFT) = prefill + 1 decode step ─────────
mx_reset()
sid_a = "bench_a"
wrapper.clear_session(sid_a); wrapper.active_session = sid_a
t0 = time.perf_counter()
_ = wrapper.generate(prompt=prompt_text, max_new_tokens=1, temperature=0.0)
mx.eval(mx.zeros(1))
ttft = time.perf_counter() - t0
wrapper.clear_session(sid_a)
peak_a = mx_peak()
print(f"[active] ttft={ttft:.3f}s", flush=True)

gc.collect(); mx.clear_cache()

# ── Pass B: full generation (prefill + GEN decode steps) ─────────────────
mx_reset()
sid_b = "bench_b"
wrapper.clear_session(sid_b); wrapper.active_session = sid_b
t0 = time.perf_counter()
response = wrapper.generate(prompt=prompt_text, max_new_tokens=GEN, temperature=0.0)
mx.eval(mx.zeros(1))
total_time = time.perf_counter() - t0
peak_b = mx_peak()
wrapper.clear_session(sid_b)
print(f"[active] total_time={total_time:.3f}s", flush=True)

# ── Derived timing ─────────────────────────────────────────────────────────
# ttft ≈ prefill + 1 decode step
# total ≈ prefill + GEN decode steps
# => decode_per_step ≈ (total - ttft) / (GEN - 1)
# => prefill_s ≈ ttft - decode_per_step
n_extra = max(1, GEN - 1)
decode_per_step = max(0.001, (total_time - ttft) / n_extra)
prefill_s = max(0.0, ttft - decode_per_step)
decode_s = decode_per_step * GEN

result = {
    "prompt_tokens": prompt_tokens, "gen_tokens": GEN,
    "prefill_s": prefill_s,
    "decode_s": decode_s,
    "decode_tps": GEN / decode_s if decode_s > 0 else None,
    "ttft_s": ttft,
    "mx_peak_gb": peak_b,
    "output_preview": response[:300],
    "needle_found": NEEDLE_PASSCODE in response,
    "engine": "active", "ctx_target": CTX, "gen_target": GEN, "status": "ok",
    "_timing_note": f"ttft={ttft:.3f} total={total_time:.3f} decode_per_step={decode_per_step:.4f}",
}

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"[active] prefill={prefill_s:.2f}s decode_tps={result['decode_tps']:.2f} "
      f"needle={'PASS' if result['needle_found'] else 'FAIL'}", flush=True)
print(f"[active] wrote {OUT_PATH}", flush=True)
'''


DENSE_WORKER = r'''
import sys, os, json, time, gc

REPO, CTX, GEN, MODEL_ID, OUT_PATH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
NEEDLE_PASSCODE = "OMEGA-7741-DELTA"

sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

def mx_peak():
    try:
        import mlx.core as mx
        for obj, name in ((mx, "get_peak_memory"), (getattr(mx,"metal",None), "get_peak_memory")):
            if obj and hasattr(obj, name):
                return float(getattr(obj, name)()) / 1e9
    except: pass
    return None

def mx_reset():
    try:
        import mlx.core as mx
        for obj, name in ((mx, "reset_peak_memory"), (getattr(mx,"metal",None), "reset_peak_memory")):
            if obj and hasattr(obj, name):
                try: getattr(obj, name)()
                except: pass
    except: pass

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load(MODEL_ID)

# Build prompt — build_niah_prompt returns (text, token_count)
try:
    from bench_common import build_niah_prompt
    _result = build_niah_prompt(CTX, tokenizer)
    prompt_text = _result[0] if isinstance(_result, tuple) else _result
except Exception:
    pfile = os.path.join(REPO, "benchmarks", "results", f"prompt_{CTX}.txt")
    if os.path.exists(pfile):
        with open(pfile) as f:
            prompt_text = f.read()
    else:
        raise

ids = tokenizer.encode(prompt_text)
prompt_tokens = len(ids)
print(f"[dense] prompt_tokens={prompt_tokens}", flush=True)

# ── Warmup: single token to compile Metal kernels ─────────────────────────
_wc = make_prompt_cache(model)
_w = model(mx.array(ids[:1])[None], cache=_wc)
mx.eval(_w)
del _wc, _w
gc.collect(); mx.clear_cache()
print("[dense] warmup done", flush=True)

# ── Measured prefill: chunked at 512 ──────────────────────────────────────
mx_reset()
cache = make_prompt_cache(model)
CH = 512
logits = None
t0 = time.perf_counter()
for cs in range(0, len(ids), CH):
    chunk = mx.array(ids[cs:cs + CH])[None]
    logits = model(chunk, cache=cache)
    mx.eval(logits)
last = logits[:, -1, :]
y = mx.argmax(last, axis=-1)
mx.eval(y)
prefill_s = time.perf_counter() - t0
print(f"[dense] prefill={prefill_s:.3f}s", flush=True)

# Release activation memory
mx.eval(); mx.clear_cache(); gc.collect()
peak_pre = mx_peak()

# ── Measured decode ────────────────────────────────────────────────────────
generated = []
t0 = time.perf_counter()
for _ in range(GEN):
    generated.append(int(y.item()))
    logits = model(y[None], cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
decode_s = time.perf_counter() - t0
peak_post = mx_peak()
print(f"[dense] decode={decode_s:.3f}s ({GEN/decode_s:.2f} tok/s)", flush=True)

text = tokenizer.decode(generated)
result = {
    "prompt_tokens": prompt_tokens, "gen_tokens": len(generated),
    "prefill_s": prefill_s,
    "decode_s": decode_s,
    "decode_tps": len(generated) / decode_s if decode_s > 0 else None,
    "ttft_s": None,
    "mx_peak_gb": peak_post,
    "output_preview": text[:300],
    "needle_found": NEEDLE_PASSCODE in text,
    "engine": "dense", "ctx_target": CTX, "gen_target": GEN, "status": "ok",
}

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"[dense] prefill={prefill_s:.2f}s decode_tps={result['decode_tps']:.2f} "
      f"needle={'PASS' if result['needle_found'] else 'FAIL'}", flush=True)
print(f"[dense] wrote {OUT_PATH}", flush=True)
'''


def run_cell(engine, ctx, gen, repo):
    out_path = os.path.join(RESULTS, f"clean_{engine}_{ctx}.json")
    code = ACTIVE_WORKER if engine == "active" else DENSE_WORKER

    print(f"\n{'='*60}", flush=True)
    print(f"  {engine.upper()} @ {ctx//1024}k  (gen={gen})", flush=True)
    print(f"{'='*60}", flush=True)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    proc = subprocess.run(
        [VENV_PY, "-c", code, repo, str(ctx), str(gen), MODEL_ID, out_path],
        env=env, cwd=repo,
    )

    if proc.returncode != 0:
        print(f"  [FAIL] {engine} @ {ctx//1024}k  (exit {proc.returncode})", flush=True)
        return False

    if os.path.exists(out_path):
        with open(out_path) as f:
            r = json.load(f)
        print(f"  ✓  prefill={r['prefill_s']:.2f}s  "
              f"decode={r['decode_tps']:.2f} tok/s  "
              f"peak={r['mx_peak_gb']:.2f} GB  "
              f"needle={'PASS' if r['needle_found'] else 'FAIL'}", flush=True)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS_DEFAULT)
    ap.add_argument("--gen", type=int, default=GEN_DEFAULT)
    ap.add_argument("--engines", nargs="+", choices=["active", "dense"],
                    default=["active", "dense"])
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.time()
    summary = {}
    for ctx in args.contexts:
        for engine in args.engines:
            ok = run_cell(engine, ctx, args.gen, REPO)
            summary[f"{engine}_{ctx}"] = "ok" if ok else "FAIL"

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE  ({elapsed:.0f}s / {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:25s}  {v}")
    print(f"\n  clean_{{engine}}_{{ctx}}.json in {RESULTS}/")


if __name__ == "__main__":
    main()
