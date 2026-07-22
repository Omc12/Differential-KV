#!/usr/bin/env python3
"""
clean_64k_v2.py — Single-pass 64k benchmark.

We already know from the aborted TTFT pass that active prefill at 64k takes ~1023s.
This script does ONE generate(128) call per engine and uses that single wall-clock
time minus the known prefill to derive decode_tps.

For active: use the measured ttft_s (1023.9s) from the aborted run as prefill reference.
  Single pass: generate(128) → total_s → decode_s = total_s - prefill_s
For dense: direct chunked prefill + decode loop (already single-pass, clean).

No warmup at 64k to avoid OOM.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
VENV_PY = os.path.join(REPO, "diffkv_venv", "bin", "python3")
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CTX = 65536
GEN = 128

# TTFT measured from the aborted run (prefill + 1 decode step at 64k)
# This IS the prefill time — 1 step is negligible vs 1023s prefill
KNOWN_TTFT_S = 1023.917

ACTIVE_W = r'''
import sys, os, json, time, gc

REPO=sys.argv[1]; MODEL_ID=sys.argv[2]; OUT=sys.argv[3]; KNOWN_TTFT=float(sys.argv[4])
CTX=65536; GEN=128
NEEDLE="OMEGA-7741-DELTA"

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

pfile = os.path.join(REPO, "benchmarks", "results", "prompt_65536.txt")
with open(pfile) as f:
    prompt_text = f.read()

prompt_tokens = len(wrapper.tokenizer.encode(prompt_text))
print(f"[active-64k] prompt_tokens={prompt_tokens}  known_ttft={KNOWN_TTFT:.2f}s", flush=True)

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

# ── Single pass: generate(128 tokens), time end-to-end ───────────────────
# prefill_s  ≈ KNOWN_TTFT (measured directly from the previous TTFT pass)
# decode_s   = total_s - KNOWN_TTFT
# decode_tps = GEN / decode_s
mx_reset()
sid = "bench64_single"
wrapper.clear_session(sid); wrapper.active_session = sid
print("[active-64k] starting single 128-token generate...", flush=True)
t0 = time.perf_counter()
response = wrapper.generate(prompt=prompt_text, max_new_tokens=GEN, temperature=0.0)
mx.eval(mx.zeros(1))
total_s = time.perf_counter() - t0
peak = mx_peak()
print(f"[active-64k] total={total_s:.3f}s  peak={peak:.2f}GB", flush=True)

# Derive decode timing using known TTFT as prefill anchor
# One decode step ≈ (total - KNOWN_TTFT) / (GEN - 1)  [subtract 1 because TTFT includes 1 step]
n_extra = max(1, GEN - 1)
decode_per_step = max(0.001, (total_s - KNOWN_TTFT) / n_extra)
prefill_s = max(0.0, KNOWN_TTFT - decode_per_step)  # TTFT = prefill + 1 step
decode_s = decode_per_step * GEN

result = {
    "prompt_tokens": prompt_tokens, "gen_tokens": GEN,
    "prefill_s": prefill_s,
    "decode_s": decode_s,
    "decode_tps": GEN / decode_s if decode_s > 0 else None,
    "ttft_s": KNOWN_TTFT,
    "mx_peak_gb": peak,
    "output_preview": response[:300],
    "needle_found": NEEDLE in response,
    "engine": "active", "ctx_target": CTX, "gen_target": GEN, "status": "ok",
    "_timing": f"known_ttft={KNOWN_TTFT:.3f} total={total_s:.3f} dps={decode_per_step:.4f}",
}
with open(OUT, "w") as f: json.dump(result, f, indent=2)
print(f"[active-64k] prefill={prefill_s:.2f}s decode_tps={result['decode_tps']:.2f} "
      f"needle={'PASS' if result['needle_found'] else 'FAIL'}", flush=True)
'''

DENSE_W = r'''
import sys, os, json, time, gc

REPO=sys.argv[1]; MODEL_ID=sys.argv[2]; OUT=sys.argv[3]
CTX=65536; GEN=128
NEEDLE="OMEGA-7741-DELTA"

sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load(MODEL_ID)

pfile = os.path.join(REPO, "benchmarks", "results", "prompt_65536.txt")
with open(pfile) as f:
    prompt_text = f.read()

ids = tokenizer.encode(prompt_text)
prompt_tokens = len(ids)
print(f"[dense-64k] prompt_tokens={prompt_tokens}", flush=True)

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

# ── Chunked prefill at CH=256 ─────────────────────────────────────────────
mx_reset()
cache = make_prompt_cache(model)
CH = 256
logits = None
t0 = time.perf_counter()
for cs in range(0, len(ids), CH):
    chunk = mx.array(ids[cs:cs+CH])[None]
    logits = model(chunk, cache=cache)
    mx.eval(logits)
last = logits[:, -1, :]
y = mx.argmax(last, axis=-1)
mx.eval(y)
prefill_s = time.perf_counter() - t0
print(f"[dense-64k] prefill={prefill_s:.3f}s", flush=True)

mx.eval(); mx.clear_cache(); gc.collect()

# ── Decode loop ────────────────────────────────────────────────────────────
generated = []
t0 = time.perf_counter()
for _ in range(GEN):
    generated.append(int(y.item()))
    logits = model(y[None], cache=cache)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
decode_s = time.perf_counter() - t0
peak = mx_peak()
print(f"[dense-64k] decode={decode_s:.3f}s ({GEN/decode_s:.2f} tok/s)", flush=True)

text = tokenizer.decode(generated)
result = {
    "prompt_tokens": prompt_tokens, "gen_tokens": len(generated),
    "prefill_s": prefill_s, "decode_s": decode_s,
    "decode_tps": len(generated)/decode_s,
    "ttft_s": None, "mx_peak_gb": peak,
    "output_preview": text[:300],
    "needle_found": NEEDLE in text,
    "engine": "dense", "ctx_target": CTX, "gen_target": GEN, "status": "ok",
}
with open(OUT, "w") as f: json.dump(result, f, indent=2)
print(f"[dense-64k] prefill={prefill_s:.2f}s decode_tps={result['decode_tps']:.2f} "
      f"needle={'PASS' if result['needle_found'] else 'FAIL'}", flush=True)
'''


def run(engine, code, extra_args=None):
    out = os.path.join(RESULTS, f"clean_{engine}_{CTX}.json")
    print(f"\n{'='*60}\n  {engine.upper()} @ 64k (single-pass)\n{'='*60}", flush=True)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    cmd = [VENV_PY, "-c", code, REPO, MODEL_ID, out]
    if extra_args:
        cmd += extra_args
    proc = subprocess.run(cmd, env=env, cwd=REPO)
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


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    ok_a = run("active", ACTIVE_W, extra_args=[str(KNOWN_TTFT_S)])
    ok_d = run("dense", DENSE_W)
    elapsed = time.time() - t0
    print(f"\n  active_65536: {'ok' if ok_a else 'FAIL'}")
    print(f"  dense_65536:  {'ok' if ok_d else 'FAIL'}")
    print(f"  Time: {elapsed:.0f}s")
