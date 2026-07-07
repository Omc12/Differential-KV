#!/usr/bin/env python3
"""Isolated single-cell measurement worker for the DiffKV paper (CLEAN protocol).

Runs ONE (engine, context) measurement in its own process so memory is
attributable and an OOM kills only this cell. Matches the ACTIVE CLI's real
configuration: preset=mid, serving_mode=balanced, rank=32, int4, with DiffKV's
sparse path FORCED ON at every context (DIFFKV_COMPRESSED_DECODE=1) so the
measurement reflects the full DiffKV path, plus the shipping decode-cache /
sparse-prefill / adaptive sparse-bias. The exact config is written into the
result JSON so every number is self-describing.

Usage:
  python cell_worker.py --engine active --ctx 16384 --prompt-file P --result-file R
  python cell_worker.py --engine dense  --ctx 16384 --prompt-file P --result-file R
"""
import os
import sys
import json
import time
import argparse
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ACTIVE_RUNTIME_DIR = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")
NEEDLE = "OMEGA-7741-DELTA"

# ── The DiffKV active configuration under test (matches the CLI: mid/balanced) ──
# Set BEFORE importing the wrapper (the manager reads these at construction).
ACTIVE_ENV = {
    "DIFFKV_COMPRESSED_DECODE": "1",   # force the sparse path at EVERY context (complete DiffKV)
    "DIFFKV_DECODE_CACHE": "1",         # decompress-and-cache fast decode (bit-exact)
    "DIFFKV_SPARSE_PREFILL": "1",       # block-sparse prefill
    "DIFFKV_SPARSE_BIAS": "auto",       # adaptive merge bias (as the CLI ships)
    "DIFFKV_MAX_RESIDUAL": "128",       # default residual budget
    "DIFFKV_ROUTER": "residual",        # residual-key router
    "DIFFKV_TOPK_BLOCKS": "16",         # top-K routed blocks
    "DIFFKV_SVD_SEED": "1234",          # deterministic SVD
    "DIFFKV_PRESET": "mid",
}
ACTIVE_CFG = {"quantization": "int4", "rank": 32, "block_size": 256,
              "micro_block_size": 256, "preset": "mid", "serving_mode": "balanced"}


def _mx_peak():
    import mlx.core as mx
    for o, n in ((mx, "get_peak_memory"), (getattr(mx, "metal", None), "get_peak_memory")):
        if o is not None and hasattr(o, n):
            try:
                return float(getattr(o, n)()) / 1e9
            except Exception:
                pass
    return None


def _mx_reset_peak():
    import mlx.core as mx
    for o, n in ((mx, "reset_peak_memory"), (getattr(mx, "metal", None), "reset_peak_memory")):
        if o is not None and hasattr(o, n):
            try:
                getattr(o, n)(); return
            except Exception:
                pass


def run_active(ctx, gen, prompt_text):
    for k, v in ACTIVE_ENV.items():
        os.environ[k] = v
    os.chdir(ACTIVE_RUNTIME_DIR)
    sys.path.insert(0, ACTIVE_RUNTIME_DIR)
    import numpy as np
    import torch
    import mlx.core as mx  # noqa: F401
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper

    _mx_reset_peak()
    wrapper = DiffKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=dict(ACTIVE_CFG))
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model
    ids = tok.encode(prompt_text)

    # Warmup (compile kernels) on a throwaway 1-token session.
    try:
        wsid = "warmup"; mgr.clear_session(wsid); wrapper._session_token_ids[wsid] = []
        mgr.init_session(wsid, prefill_len=1)
        mgr.register_prefill_tokens(wsid, torch.tensor([ids[0]], dtype=torch.long))
        model._diffkv_session_ids = [wsid]
        _w = model(torch.tensor([[ids[0]]], dtype=torch.long), torch.tensor([[0]], dtype=torch.long))
        _ = _w.logits[0, -1].cpu().numpy(); mgr.clear_session(wsid)
    except Exception:
        pass

    sid = "cell"; mgr.clear_session(sid); wrapper._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    model._diffkv_session_ids = [sid]

    CH = 512
    output = None
    t0 = time.perf_counter()
    for cs in range(0, len(ids), CH):
        chunk = ids[cs:cs + CH]
        ct = torch.tensor([chunk], dtype=torch.long)
        pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
        output = model(ct, pt)
        mgr.compress_deferred_prefill_blocks(sid)
    logits = output.logits[0, -1].cpu().numpy()
    prefill_s = time.perf_counter() - t0

    cur = len(ids); generated = []
    t0 = time.perf_counter()
    for _ in range(gen):
        nid = int(np.argmax(logits)); generated.append(nid)
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        output = model(torch.tensor([[nid]], dtype=torch.long), torch.tensor([[cur]], dtype=torch.long))
        logits = output.logits[0, -1].cpu().numpy(); cur += 1
    decode_s = time.perf_counter() - t0
    text = tok.decode(generated)
    return {
        "prompt_tokens": len(ids), "gen_tokens": len(generated),
        "prefill_s": prefill_s, "decode_s": decode_s,
        "decode_tps": len(generated) / decode_s if decode_s > 0 else None,
        "mx_peak_gb": _mx_peak(), "output_preview": text[:200],
        "needle_found": NEEDLE in text,
        "config": {"engine": "active", **ACTIVE_CFG, **ACTIVE_ENV},
    }


def run_dense(ctx, gen, prompt_text):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    _mx_reset_peak()
    model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ids = tokenizer.encode(prompt_text)

    _wc = make_prompt_cache(model); _w = model(mx.array(ids[:1])[None], cache=_wc)
    mx.eval(_w); del _wc, _w

    cache = make_prompt_cache(model); CH = 512; logits = None
    t0 = time.perf_counter()
    for cs in range(0, len(ids), CH):
        logits = model(mx.array(ids[cs:cs + CH])[None], cache=cache); mx.eval(logits)
    y = mx.argmax(logits[:, -1, :], axis=-1); mx.eval(y)
    prefill_s = time.perf_counter() - t0

    import gc
    mx.eval(); mx.clear_cache(); gc.collect()
    generated = []; t0 = time.perf_counter()
    for _ in range(gen):
        generated.append(int(y.item()))
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1); mx.eval(y)
    decode_s = time.perf_counter() - t0
    text = tokenizer.decode(generated)
    return {
        "prompt_tokens": len(ids), "gen_tokens": len(generated),
        "prefill_s": prefill_s, "decode_s": decode_s,
        "decode_tps": len(generated) / decode_s if decode_s > 0 else None,
        "mx_peak_gb": _mx_peak(), "output_preview": text[:200],
        "needle_found": NEEDLE in text,
        "config": {"engine": "dense", "quantization": "int4", "full_kv": True},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["active", "dense"])
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--result-file", required=True)
    a = ap.parse_args()
    a.prompt_file = os.path.abspath(a.prompt_file)
    a.result_file = os.path.abspath(a.result_file)
    prompt = open(a.prompt_file, encoding="utf-8").read()
    t = time.perf_counter()
    try:
        res = (run_active if a.engine == "active" else run_dense)(a.ctx, a.gen, prompt)
        res["status"] = "ok"
    except Exception as e:
        res = {"status": "error", "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-2000:]}
    res.update({"engine": a.engine, "ctx_target": a.ctx, "gen_target": a.gen,
                "worker_wall_s": time.perf_counter() - t})
    with open(a.result_file, "w") as f:
        json.dump(res, f, indent=2)
    print("__CELL__ " + json.dumps({k: v for k, v in res.items()
                                    if k not in ("traceback", "config")}))


if __name__ == "__main__":
    main()
