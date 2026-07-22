#!/usr/bin/env python3
"""
Single-measurement worker for the DiffKV multi-engine benchmark.

Runs ONE (engine, context_length) measurement in an isolated process so that
(a) RAM is attributable to exactly one engine, and (b) an OOM kills only this
worker, not the orchestrator. Writes a JSON result to --result-file on success.

Engines
-------
  native : C++ DiffKV binary (GGUF Q4_K_M) driven over its stdin/stdout protocol.
           Prefill seconds come from the binary's own [PREFILL_TIME] stderr line;
           decode token count from [Timing Step N] lines (DIFFKV_TIME_DECODE=1).
  active : DiffKV ACTIVE_RUNTIME — MLXDiffKVWrapper (int4), the reference DiffKV
           impl. Compressed KV. Prefill + decode timed with perf_counter / mx.eval.
  dense  : Plain mlx_lm (same int4 weights, same engine) with a FULL KV cache —
           the no-compression control. Isolates exactly what DiffKV buys.
  ollama : llama.cpp via the ollama server (GGUF Q4_K_M). Exact prefill/decode
           timings + token counts from /api/generate (prompt_eval_*/eval_*).
           Its RAM is measured externally by the orchestrator (server process).

Decode is greedy (temperature 0) and runs a FIXED number of tokens (EOS ignored)
for native/active/dense so TPS is directly comparable. ollama reports its own
eval_count.
"""

import os
import sys
import json
import time
import argparse
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE_RUNTIME_DIR = os.path.join(REPO, "ACTIVE_RUNTIME")

sys.path.insert(0, HERE)
from bench_common import NEEDLE_PASSCODE  # noqa: E402


def _mx_peak_gb():
    try:
        import mlx.core as mx
    except Exception:
        return None
    for obj, name in ((mx, "get_peak_memory"),
                      (getattr(mx, "metal", None), "get_peak_memory")):
        if obj is not None and hasattr(obj, name):
            try:
                return float(getattr(obj, name)()) / 1e9
            except Exception:
                pass
    return None


def _mx_reset_peak():
    try:
        import mlx.core as mx
    except Exception:
        return
    for obj, name in ((mx, "reset_peak_memory"),
                      (getattr(mx, "metal", None), "reset_peak_memory")):
        if obj is not None and hasattr(obj, name):
            try:
                getattr(obj, name)()
                return
            except Exception:
                pass


# ───────────────────────────── native (C++ DiffKV) ──────────────────────────
def run_native(args, prompt_text):
    import re
    import threading
    import subprocess

    env = os.environ.copy()
    # Single-threaded BLAS (see cli.py rationale: keeps the Mac responsive during
    # the SVD-heavy prefill and avoids core starvation).
    env.update({
        "DIFFKV_MAX_CTX_TK": str(args.ctx + args.gen + 512),
        "DIFFKV_MICRO_BLOCK_SIZE": "256",
        "DIFFKV_PREFILL_CHUNK_SIZE": "512",
        "DIFFKV_MAX_TOKENS": str(args.gen),
        "DIFFKV_USE_GPU": "1",
        "DIFFKV_NATIVE_ATTN": "1",
        "DIFFKV_TEMPERATURE": "0.0",
        "DIFFKV_TOP_P": "1.0",
        "DIFFKV_REPETITION_PENALTY": "1.0",
        "DIFFKV_DBG_PREFILL_TIME": "1",
        "DIFFKV_TIME_DECODE": "1",
        "DIFFKV_MPS_APPROXIMATE_ATTN": "1",
        "DIFFKV_COMPRESSOR_THREADS": "4",
        "DIFFKV_ENABLE_FACTUAL": "1",
    })

    proc = subprocess.Popen(
        [args.native_binary, args.native_model, "-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, env=env,
    )
    # report child pid so the orchestrator can also track it directly
    if args.pid_file:
        with open(args.pid_file, "w") as f:
            f.write(str(proc.pid))

    stderr_chunks = []

    def _read_err():
        try:
            for line in iter(proc.stderr.readline, b""):
                stderr_chunks.append(line.decode("utf-8", "replace"))
        except Exception:
            pass

    et = threading.Thread(target=_read_err, daemon=True)
    et.start()

    READY, RESP, FIN = b"__READY__", b"__RESPONSE__", b"__FINISH__"

    def _read_until(token, ctx_msg):
        buf = b""
        while token not in buf:
            c = os.read(proc.stdout.fileno(), 65536)
            if not c:
                raise RuntimeError(
                    f"native process exited before {ctx_msg}; stderr tail:\n"
                    + "".join(stderr_chunks[-40:]))
            buf += c
        return buf

    # Warmup runs while we drain to __READY__.
    _read_until(READY, "__READY__")

    single = prompt_text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
    t_send = time.perf_counter()
    proc.stdin.write((single + "\n").encode("utf-8"))
    proc.stdin.flush()

    buf = _read_until(RESP, "__RESPONSE__")
    after = buf.split(RESP, 1)[1]
    if after.startswith(b"\n"):
        after = after[1:]
    t_resp = time.perf_counter()

    resp = after
    t_first = t_resp if resp.strip() else None
    while FIN not in resp:
        c = os.read(proc.stdout.fileno(), 65536)
        if not c:
            break
        if t_first is None and c.strip():
            t_first = time.perf_counter()
        resp += c
    t_fin = time.perf_counter()

    text = resp.split(FIN, 1)[0].decode("utf-8", "replace").strip()

    try:
        proc.stdin.write(b"exit\n")
        proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    err = "".join(stderr_chunks)
    with open("benchmarks/results/stderr_native.log", "w") as f_err:
        f_err.write(err)
    m = re.search(r"\[PREFILL_TIME\]\s+L=(\d+).*?TOTAL=([0-9.]+)s", err, re.S)
    prompt_tokens = int(m.group(1)) if m else None
    prefill_s = float(m.group(2)) if m else (t_resp - t_send)

    steps = re.findall(r"\[Timing Step (\d+)\]", err)
    gen_tokens = (max(int(s) for s in steps) + 1) if steps else None
    if not gen_tokens:
        gen_tokens = max(1, len(text) // 4)  # rough fallback only

    decode_s = t_fin - (t_first if t_first else t_resp)
    decode_tps = gen_tokens / decode_s if decode_s > 0 else None

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
        "ttft_s": (t_first - t_send) if t_first else None,
        "mx_peak_gb": None,
        "output_preview": text[:200],
        "needle_found": NEEDLE_PASSCODE in text,
    }


# ────────────────────────── active (DiffKV MLX int4) ─────────────────────────
def run_active(args, prompt_text):
    # ── MLX path (Apple Silicon) ──────────────────────────────────────────────
    # Uses MLXDiffKVWrapper — the real on-device sparse-KV runtime.
    # DiffKVHFWrapper is the CUDA path and requires diffkv_core; do NOT use it
    # on macOS. The MLX wrapper's generate() runs prefill+decode end-to-end;
    # we time each phase by splitting into a 1-token prefill pass then N-token
    # decode pass, with mx.eval() barriers to force GPU sync before timing.
    sys.path.insert(0, ACTIVE_RUNTIME_DIR)
    import mlx.core as mx
    from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper

    _mx_reset_peak()

    # Paper config: rank-32, max_residual=128, compressed decode
    os.environ.setdefault("DIFFKV_COMPRESSED_DECODE", "1")
    os.environ.setdefault("DIFFKV_MAX_RESIDUAL", "128")
    os.environ.setdefault("DIFFKV_SPARSE_PREFILL", "1")
    os.environ.setdefault("DIFFKV_DECODE_CACHE", "1")
    os.environ.setdefault("DIFFKV_SPARSE_BIAS", "auto")
    os.environ.setdefault("DIFFKV_SEED", "1234")

    wrapper = MLXDiffKVWrapper(
        model_id=args.dense_model_id,  # mlx-community/Qwen2.5-1.5B-Instruct-4bit
        config={"rank": 32, "block_size": 256},
    )
    wrapper.ensure_loaded()
    tok = wrapper.tokenizer
    ids = tok.encode(prompt_text)
    prompt_tokens = len(ids)

    # ── Pass 1: time prompt+1 token → prefill_s, then measure decode separately ──
    # Strategy: run generate(1 token) → gives prefill_s ≈ prefill + 1 step.
    # Then run generate(args.gen tokens) → total_s = prefill + all decode steps.
    # decode_per_tok ≈ (total_s - prefill_s) / (args.gen - 1)  [difference cancels prefill]
    # This avoids any hardcoded proportions.

    # Pass 1 — 1 token (prefill + 1 decode step)
    sid1 = "bench_p1"
    wrapper.manager.clear_session(sid1)
    if hasattr(wrapper, "_session_token_ids"):
        wrapper._session_token_ids[sid1] = []
    wrapper.active_session = sid1
    _mx_reset_peak()
    t0 = time.perf_counter()
    _ = wrapper.generate(prompt=prompt_text, max_new_tokens=1, temperature=0.0)
    mx.eval(mx.zeros(1))
    time_1tok = time.perf_counter() - t0

    # Pass 2 — args.gen tokens (prefill + args.gen decode steps)
    sid2 = "bench_p2"
    wrapper.manager.clear_session(sid2)
    if hasattr(wrapper, "_session_token_ids"):
        wrapper._session_token_ids[sid2] = []
    wrapper.active_session = sid2
    t0 = time.perf_counter()
    response = wrapper.generate(prompt=prompt_text, max_new_tokens=args.gen, temperature=0.0)
    mx.eval(mx.zeros(1))
    time_ntok = time.perf_counter() - t0

    # Derived timing: difference cancels the common prefill cost
    # decode_per_step ≈ (time_ntok - time_1tok) / (args.gen - 1)
    n_extra = max(1, args.gen - 1)
    decode_per_step = max(0.001, (time_ntok - time_1tok) / n_extra)
    decode_s = decode_per_step * args.gen
    prefill_s = time_1tok - decode_per_step  # prefill = 1-tok total minus 1 decode step

    gen_toks = args.gen
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_toks,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": gen_toks / decode_s if decode_s > 0 else None,
        "ttft_s": None,
        "mx_peak_gb": _mx_peak_gb(),
        "output_preview": response[-300:] if len(response) > 300 else response,
        "needle_found": NEEDLE_PASSCODE in response,
    }



# ──────────────────── dense (plain mlx_lm int4, full KV) ─────────────────────
def run_dense(args, prompt_text):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    _mx_reset_peak()
    model, tokenizer = load(args.dense_model_id)
    ids = tokenizer.encode(prompt_text)
    prompt_tokens = len(ids)

    # Warmup: compile Metal kernels with a throwaway 1-token forward so the
    # timed prefill below measures compute, not one-shot kernel compilation.
    _wc = make_prompt_cache(model)
    _w = model(mx.array(ids[:1])[None], cache=_wc)
    mx.eval(_w)
    del _wc, _w

    cache = make_prompt_cache(model)
    CH = 512
    if os.environ.get("DIFFKV_PREFILL_CHUNK_SIZE"):
        try:
            CH = int(os.environ["DIFFKV_PREFILL_CHUNK_SIZE"])
        except ValueError:
            pass
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

    generated = []
    # ── Prefill → Decode boundary: release peak activation memory ──────────────
    # Mirrors what the active runtime already does at this boundary.
    # Without this, dense peak memory is artificially inflated in benchmarks.
    import gc as _gc
    mx.eval()
    mx.clear_cache()
    _gc.collect()
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(args.gen):
        generated.append(int(y.item()))
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
    decode_s = time.perf_counter() - t0

    text = tokenizer.decode(generated)
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": len(generated),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": len(generated) / decode_s if decode_s > 0 else None,
        "ttft_s": None,
        "mx_peak_gb": _mx_peak_gb(),
        "output_preview": text[:200],
        "needle_found": NEEDLE_PASSCODE in text,
    }


# ─────────────────────────── ollama (llama.cpp) ──────────────────────────────
def run_ollama(args, prompt_text):
    import urllib.request

    base = args.ollama_url.rstrip("/")
    payload = {
        "model": args.ollama_model,
        "prompt": prompt_text,
        "stream": False,
        "raw": True,  # we already applied the chat template; do not double-apply
        "keep_alive": 30,  # keep loaded briefly so we can read /api/ps below
        "options": {
            "temperature": 0,
            "num_predict": args.gen,
            "num_ctx": args.ctx + args.gen + 256,
        },
    }
    # Warmup: load the model + allocate the context once (we unload between
    # cells, so without this every cell would pay a cold mmap fault-in, inflating
    # prefill and decode). num_ctx must match so the KV allocation is identical.
    try:
        warm = {"model": args.ollama_model, "prompt": "hello", "stream": False,
                "raw": True, "keep_alive": 30,
                "options": {"temperature": 0, "num_predict": 1,
                            "num_ctx": args.ctx + args.gen + 256}}
        urllib.request.urlopen(urllib.request.Request(
            base + "/api/generate", data=json.dumps(warm).encode(),
            headers={"Content-Type": "application/json"}), timeout=args.http_timeout).read()
    except Exception:
        pass

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + "/api/generate", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.http_timeout) as r:
        res = json.loads(r.read().decode("utf-8"))

    if res.get("error"):
        raise RuntimeError(f"ollama error: {res['error']}")

    # Authoritative per-run memory from ollama itself (model + KV), then unload.
    size_vram_gb = size_gb = None
    try:
        with urllib.request.urlopen(base + "/api/ps", timeout=30) as r:
            ps = json.loads(r.read().decode("utf-8"))
        for m in ps.get("models", []):
            if m.get("model") == args.ollama_model or m.get("name") == args.ollama_model:
                size_vram_gb = (m.get("size_vram") or 0) / 1e9
                size_gb = (m.get("size") or 0) / 1e9
                break
    except Exception:
        pass
    try:
        urllib.request.urlopen(urllib.request.Request(
            base + "/api/generate",
            data=json.dumps({"model": args.ollama_model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"}), timeout=30).read()
    except Exception:
        pass

    pe_count = res.get("prompt_eval_count")
    pe_ns = res.get("prompt_eval_duration") or 0
    ev_count = res.get("eval_count") or 0
    ev_ns = res.get("eval_duration") or 0
    text = res.get("response", "")

    prefill_s = (pe_ns / 1e9) if pe_ns else None
    decode_s = (ev_ns / 1e9) if ev_ns else None
    return {
        "prompt_tokens": pe_count,
        "gen_tokens": ev_count,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": (ev_count / decode_s) if decode_s else None,
        "ttft_s": None,
        "mx_peak_gb": None,  # filled externally by orchestrator (server tree mem)
        "ollama_size_vram_gb": size_vram_gb,  # ollama's own model+KV report
        "ollama_size_gb": size_gb,
        "output_preview": text[:200],
        "needle_found": NEEDLE_PASSCODE in text,
    }


RUNNERS = {
    "native": run_native,
    "active": run_active,
    "dense": run_dense,
    "ollama": run_ollama,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(RUNNERS))
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--result-file", required=True)
    ap.add_argument("--pid-file", default=None)
    ap.add_argument("--native-binary",
                    default=os.path.join(REPO, "diffkv_native/build/diffkv_native"))
    ap.add_argument("--native-model",
                    default=os.path.join(REPO, "diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
    ap.add_argument("--dense-model-id", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--ollama-model", default="qwen2.5:1.5b-instruct")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--http-timeout", type=float, default=3600.0)
    args = ap.parse_args()

    # Resolve to absolute paths now: some engine runners chdir (e.g. active ->
    # ACTIVE_RUNTIME), which would otherwise break these relative paths.
    args.prompt_file = os.path.abspath(args.prompt_file)
    args.result_file = os.path.abspath(args.result_file)
    if args.pid_file:
        args.pid_file = os.path.abspath(args.pid_file)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    t_start = time.perf_counter()
    try:
        result = RUNNERS[args.engine](args, prompt_text)
        result["status"] = "ok"
    except Exception as e:
        result = {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
    result["engine"] = args.engine
    result["ctx_target"] = args.ctx
    result["gen_target"] = args.gen
    result["worker_wall_s"] = time.perf_counter() - t_start

    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # Also echo for live logs.
    print("\n__BENCH_RESULT__ " + json.dumps(
        {k: v for k, v in result.items() if k != "traceback"}))


if __name__ == "__main__":
    main()
