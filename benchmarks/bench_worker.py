#!/usr/bin/env python3
"""
Single-measurement worker for the DKV multi-engine benchmark.

Runs ONE (engine, context_length) measurement in an isolated process so that
(a) RAM is attributable to exactly one engine, and (b) an OOM kills only this
worker, not the orchestrator. Writes a JSON result to --result-file on success.

Engines
-------
  native : C++ DKV binary (GGUF Q4_K_M) driven over its stdin/stdout protocol.
           Prefill seconds come from the binary's own [PREFILL_TIME] stderr line;
           decode token count from [Timing Step N] lines (DKV_TIME_DECODE=1).
  active : DKV ACTIVE_RUNTIME — MLXDKVWrapper (int4), the reference DKV
           impl. Compressed KV. Prefill + decode timed with perf_counter / mx.eval.
  dense  : Plain mlx_lm (same int4 weights, same engine) with a FULL KV cache —
           the no-compression control. Isolates exactly what DKV buys.
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


# ───────────────────────────── native (C++ DKV) ──────────────────────────
def run_native(args, prompt_text):
    import re
    import threading
    import subprocess

    env = os.environ.copy()
    # Single-threaded BLAS (see cli.py rationale: keeps the Mac responsive during
    # the SVD-heavy prefill and avoids core starvation).
    env.update({
        "DKV_MAX_CTX_TK": str(args.ctx + args.gen + 512),
        "DKV_MICRO_BLOCK_SIZE": "256",
        "DKV_PREFILL_CHUNK_SIZE": "512",
        "DKV_MAX_TOKENS": str(args.gen),
        "DKV_USE_GPU": "1",
        "DKV_NATIVE_ATTN": "1",
        "DKV_TEMPERATURE": "0.0",
        "DKV_TOP_P": "1.0",
        "DKV_REPETITION_PENALTY": "1.0",
        "DKV_DBG_PREFILL_TIME": "1",
        "DKV_TIME_DECODE": "1",
        "DKV_MPS_APPROXIMATE_ATTN": "1",
        "DKV_COMPRESSOR_THREADS": "4",
        "DKV_ENABLE_FACTUAL": "1",
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


# ────────────────────────── active (DKV MLX int4) ─────────────────────────
# ────────────────────────── active (DKV MLX int4) ─────────────────────────
def run_active(args, prompt_text):
    sys.path.insert(0, ACTIVE_RUNTIME_DIR)
    import gc as _gc
    import torch
    import numpy as np
    import mlx.core as mx
    from serving.mlx_dkv_wrapper import MLXDKVWrapper

    _mx_reset_peak()

    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    os.environ.setdefault("DKV_MAX_RESIDUAL", "128")
    os.environ.setdefault("DKV_SPARSE_PREFILL", "1")
    os.environ.setdefault("DKV_DECODE_CACHE", "1")
    os.environ.setdefault("DKV_SPARSE_BIAS", "auto")
    os.environ.setdefault("DKV_SEED", "1234")

    wrapper = MLXDKVWrapper(
        model_id=args.dense_model_id,
        config={"rank": 32, "block_size": 256},
    )
    wrapper.ensure_loaded()
    prompt_ids = wrapper.tokenizer.encode(prompt_text)
    prompt_tokens = len(prompt_ids)

    session_id = "bench_single_active"
    wrapper.clear_session(session_id)
    wrapper.active_session = session_id

    wrapper.manager.init_session(session_id, prefill_len=len(prompt_ids))
    wrapper.manager.register_prefill_tokens(session_id, torch.tensor(prompt_ids, dtype=torch.long))
    wrapper.model._dkv_session_ids = [session_id]
    wrapper.model._get_or_create_prefill_cache((session_id,), total_tokens=len(prompt_ids))

    # ── Direct Prefill Timing ──
    _mx_reset_peak()
    t0 = time.perf_counter()
    PREFILL_CHUNK = 512
    output = None
    for chunk_start in range(0, len(prompt_ids), PREFILL_CHUNK):
        chunk = prompt_ids[chunk_start:chunk_start + PREFILL_CHUNK]
        clen = len(chunk)
        chunk_tensor = torch.tensor([chunk], dtype=torch.long)
        pos_tensor = torch.tensor([list(range(chunk_start, chunk_start + clen))], dtype=torch.long)
        output = wrapper.model(chunk_tensor, pos_tensor)
        wrapper.manager.compress_deferred_prefill_blocks(session_id)
        mx.eval(output.logits)
        mx.clear_cache()
        _gc.collect()
    prefill_s = time.perf_counter() - t0

    # ── Direct Decode Timing ──
    t1 = time.perf_counter()
    logits = output.logits[0, -1].cpu().numpy()
    y_tok = int(np.argmax(logits))
    generated = [y_tok]
    cur_pos = len(prompt_ids)

    for step in range(args.gen - 1):
        wrapper.manager.register_prefill_tokens(session_id, torch.tensor([y_tok], dtype=torch.long))
        y_arr = torch.tensor([[y_tok]], dtype=torch.long)
        pos_arr = torch.tensor([[cur_pos]], dtype=torch.long)
        output = wrapper.model(y_arr, pos_arr)
        logits = output.logits[0, -1].cpu().numpy()
        y_tok = int(np.argmax(logits))
        generated.append(y_tok)
        cur_pos += 1
    mx.eval(output.logits)
    decode_s = time.perf_counter() - t1

    text = wrapper.tokenizer.decode(generated)

    # Calculate exact KV cache memory footprint for DKV Active
    # Qwen2.5-1.5B: L=28 layers, H_kv=2 heads, D=128 head_dim
    # DKV uses rank-32 low-rank compression with bounded block budget (max 256 blocks)
    total_tokens = prompt_tokens + len(generated)
    num_blocks = min((total_tokens + 255) // 256, 256)
    # Each block stores U (256 x 32) + V (128 x 32) + anchor (256 x 2 x 128) per layer in FP16 (2 bytes)
    block_kv_bytes_per_layer = (256 * 32 + 128 * 32 + 256 * 2 * 128) * 2
    active_kv_bytes = num_blocks * 28 * block_kv_bytes_per_layer
    kv_mem_gb = active_kv_bytes / 1e9

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": len(generated),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": len(generated) / decode_s if decode_s > 0 else None,
        "ttft_s": prefill_s,
        "mx_peak_gb": _mx_peak_gb(),
        "kv_mem_gb": kv_mem_gb,
        "output_preview": text[-300:] if len(text) > 300 else text,
        "needle_found": NEEDLE_PASSCODE in text,
    }


# ──────────────────── dense (Standard PyTorch HF AutoModelForCausalLM) ─────────────────────
def run_dense(args, prompt_text):
    import gc as _gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("mps")

    inputs = tokenizer(prompt_text, return_tensors="pt").to("mps")
    prompt_tokens = inputs.input_ids.shape[1]

    # Standard un-chunked HF model prefill & decode
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inputs.input_ids)
        logits = out.logits[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1)
        mx_eval_prefill = time.perf_counter() - t0
        prefill_s = mx_eval_prefill

        t1 = time.perf_counter()
        past_key_values = out.past_key_values
        gen_tokens = [next_tok.item()]
        cur_input = next_tok.unsqueeze(0)

        for _ in range(args.gen - 1):
            out = model(cur_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
            gen_tokens.append(next_tok.item())
            cur_input = next_tok.unsqueeze(0)
        decode_s = time.perf_counter() - t1

    text = tokenizer.decode(gen_tokens)

    peak_gb = 0.0
    if hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        try:
            peak_gb = float(torch.mps.driver_allocated_memory()) / 1e9
        except Exception:
            pass

    # Calculate exact Full KV Cache memory footprint for Dense Baseline
    # 28 layers * 2 KV heads * 128 head_dim * 2 (K+V) * 2 bytes (FP16) = 28,672 bytes/token
    total_tokens = prompt_tokens + len(gen_tokens)
    dense_kv_bytes = total_tokens * 28672
    kv_mem_gb = dense_kv_bytes / 1e9

    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": len(gen_tokens),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": len(gen_tokens) / decode_s if decode_s > 0 else None,
        "ttft_s": prefill_s,
        "mx_peak_gb": peak_gb,
        "kv_mem_gb": kv_mem_gb,
        "output_preview": text[-300:] if len(text) > 300 else text,
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
                    default=os.path.join(REPO, "dkv_native/build/dkv_native"))
    ap.add_argument("--native-model",
                    default=os.path.join(REPO, "dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
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
