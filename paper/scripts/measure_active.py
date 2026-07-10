#!/usr/bin/env python3
"""Clean, instrumented re-measurement of the DiffKV ACTIVE runtime for the paper.

For each context length and each decode mode it records, from INSIDE the worker
process (authoritative MLX metrics):

  * prefill_s, decode_tps, gen_tokens, needle_found, output_preview
  * mx_peak_gb            : global MLX allocator peak (prefill-dominated)
  * mx_decode_peak_gb     : MLX peak DURING DECODE only (peak reset at the
                            prefill->decode boundary) -> steady-state decode mem
  * mx_active_end_gb      : MLX live/active memory at end of decode
  * kv_store_bytes        : analytic DiffKV store footprint (compressed pool + dense)
  * kv_dense_full_bytes   : analytic full-KV cache footprint for the same seq_len
  * num_blocks, dense_len : store occupancy at end (per layer 0; ×L for totals)

Modes:
  compressed : DIFFKV_COMPRESSED_DECODE=1  (the real DiffKV sparse decode)
  exact      : DIFFKV_COMPRESSED_DECODE=0  (full-KV decode; upper-bound ablation)

Usage:
  diffkv_venv/bin/python3 paper/scripts/measure_active.py --ctx 4096 8192 16384 32768 65536 \
       --modes compressed exact --gen 128 --out paper/generated/active_modes_sweep.json
"""
import os, sys, json, time, argparse, gc

# Ensure user-site packages are in the python path for remote kernels
USER_SITE = os.path.expanduser("~/.local/lib/python3.11/site-packages")
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")
NEEDLE = "OMEGA-7741-DELTA"

# Add paths to sys.path so subprocesses can find everything (including compiled C++ modules)
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
diffkv_core_path = os.path.join(ACTIVE, "native_core", "diffkv_core")
if diffkv_core_path not in sys.path:
    sys.path.insert(0, diffkv_core_path)


import torch

def _get_peak_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    try:
        import mlx.core as mx
        for o, n in ((mx, "get_peak_memory"), (getattr(mx, "metal", None), "get_peak_memory")):
            if o is not None and hasattr(o, n):
                return float(getattr(o, n)()) / 1e9
    except Exception:
        pass
    return 0.0


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        import mlx.core as mx
        for o, n in ((mx, "reset_peak_memory"), (getattr(mx, "metal", None), "reset_peak_memory")):
            if o is not None and hasattr(o, n):
                getattr(o, n)()
                return
    except Exception:
        pass


def _get_active_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    try:
        import mlx.core as mx
        for o, n in ((mx, "get_active_memory"), (getattr(mx, "metal", None), "get_active_memory")):
            if o is not None and hasattr(o, n):
                return float(getattr(o, n)()) / 1e9
    except Exception:
        pass
    return 0.0


def analytic_kv_bytes(mgr, seq_len):
    """DiffKV store footprint (fixed pre-allocated pool + dense) vs full-KV, in bytes.

    Pools are pre-allocated at max_blocks, so the store footprint is BOUNDED &
    context-independent; we report both the allocated (real) and used footprint.

    IMPORTANT: every block also keeps `max_residual` EXACT fp16 K/V residual tokens
    (res_k/res_v) and element-wise key min/max (min_k/max_k) — these dominate the
    per-block bytes and MUST be counted. Omitting them (as an earlier version did)
    overstates the compression ratio ~3.5x (10x vs the true ~2.85x at defaults)."""
    L = mgr.num_layers; Hkv = mgr.kv_heads; d = mgr.head_dim
    B = mgr.block_size; r = mgr.rank; M = mgr.max_blocks; Dmax = mgr.max_dense_len
    R = mgr.max_residual
    fp16 = 2
    kv_tok = Hkv * d * fp16 * 2            # one exact K+V token, all kv-heads
    lowrank_block = ((B - 1) * r * fp16        # U
                     + 2 * Hkv * r * d * fp16  # V_K, V_V
                     + 2 * Hkv * d * fp16      # anchor_k, anchor_v
                     + 2 * Hkv * d * fp16      # min_k, max_k (router)
                     + 8)                       # scale fp32 + seq_len int32
    residual_block_max = R * kv_tok            # res_k + res_v at max_residual
    per_block = lowrank_block + residual_block_max
    dense_alloc = Dmax * kv_tok                # K,V fp16 over full dense buffer
    s0 = mgr.sessions.get("bench")
    nb = s0["num_blocks"][0] if s0 else 0
    dl = s0["dense_lens"][0] if s0 else 0
    # actual residual occupancy (top-by-error count per block; layer 0, ×L)
    res_n0 = s0["comp_res_n"][0][:nb] if s0 else []
    res_tokens_used = int(sum(res_n0))
    store_alloc = L * (M * per_block + dense_alloc)
    store_used = L * (nb * lowrank_block + res_tokens_used * kv_tok + dl * kv_tok)
    dense_full = L * seq_len * kv_tok
    return {
        "per_block_bytes": per_block,
        "lowrank_block_bytes": lowrank_block,
        "residual_block_bytes_max": residual_block_max,
        "store_alloc_bytes": store_alloc,
        "store_used_bytes": store_used,
        "dense_full_bytes": dense_full,
        "num_blocks_layer0": int(nb),
        "dense_len_layer0": int(dl),
        "res_tokens_layer0": res_tokens_used,
        "ratio_used_vs_dense": (dense_full / store_used) if store_used else None,
    }


def run_cell(ctx, gen, prompt_text, model_id):
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    os.environ["DIFFKV_FACTUAL_STORE"] = "0"

    is_compressed = os.environ.get("DIFFKV_COMPRESSED_DECODE", "1") != "0"
    _reset_peak()

    if is_compressed:
        from serving.hf_diffkv_wrapper import DiffKVHFWrapper
        cfg = {"quantization": None, "rank": 32, "block_size": 256,
               "micro_block_size": 256, "preset": "mid", "serving_mode": "balanced"}
        w = DiffKVHFWrapper(model_id=model_id, config=cfg, torch_dtype=torch.float16)
        w.ensure_loaded()
        tok, mgr, model = w.tokenizer, w.manager, w.model
        dev = w.device

        ids = tok.encode(prompt_text)

        # warmup (compile kernels)
        try:
            if not hasattr(w, "_session_token_ids"):
                w._session_token_ids = {}
            mgr.clear_session("warm"); w._session_token_ids["warm"] = []
            mgr.init_session("warm", prefill_len=1)
            mgr.register_prefill_tokens("warm", torch.tensor([ids[0]], dtype=torch.long, device=dev))
            model._diffkv_session_ids = ["warm"]
            _ = model(torch.tensor([[ids[0]]], device=dev), torch.tensor([[0]], device=dev)).logits[0, -1].float().cpu().numpy()
            mgr.clear_session("warm")
        except Exception:
            pass

        sid = "bench"; mgr.clear_session(sid)
        if not hasattr(w, "_session_token_ids"):
            w._session_token_ids = {}
        w._session_token_ids[sid] = []
        mgr.init_session(sid, prefill_len=len(ids))
        mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=dev))
        model._diffkv_session_ids = [sid]

        CH = 512; out = None
        t0 = time.perf_counter()
        for cs in range(0, len(ids), CH):
            ch = ids[cs:cs + CH]
            out = model(torch.tensor([ch], device=dev), torch.tensor([list(range(cs, cs + len(ch)))], device=dev))
            mgr.compress_deferred_prefill_blocks(sid)
        logits = out.logits[0, -1].float().cpu().numpy()
        prefill_s = time.perf_counter() - t0
    else:
        # Load standard un-patched Transformers model
        from transformers import BitsAndBytesConfig as _BnBConfig
        quantization_config = None
        _quant = os.environ.get("DIFFKV_QUANTIZATION")
        if _quant == "nf4":
            quantization_config = _BnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        dev = next(model.parameters()).device
        mgr = None

        ids = tok.encode(prompt_text)

        # Standard PyTorch eager baseline
        t0 = time.perf_counter()
        past_key_values = None
        CH = 512; out = None
        for cs in range(0, len(ids), CH):
            ch = ids[cs:cs + CH]
            pos = torch.tensor([list(range(cs, cs + len(ch)))], device=dev)
            out = model(torch.tensor([ch], device=dev), position_ids=pos, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
        logits = out.logits[0, -1].float().cpu().numpy()
        prefill_s = time.perf_counter() - t0

    mx_peak_prefill = _get_peak_gb()
    # ── isolate decode-phase memory: reset peak at the boundary ──
    _reset_peak()

    cur = len(ids); gen_ids = []
    t0 = time.perf_counter()
    if is_compressed:
        for _ in range(gen):
            nid = int(np.argmax(logits)); gen_ids.append(nid)
            mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long, device=dev))
            out = model(torch.tensor([[nid]], device=dev), torch.tensor([[cur]], device=dev))
            logits = out.logits[0, -1].float().cpu().numpy(); cur += 1
    else:
        for _ in range(gen):
            nid = int(np.argmax(logits)); gen_ids.append(nid)
            pos = torch.tensor([[cur]], device=dev)
            out = model(torch.tensor([[nid]], device=dev), position_ids=pos, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[0, -1].float().cpu().numpy(); cur += 1
    decode_s = time.perf_counter() - t0

    text = tok.decode(gen_ids)

    if is_compressed:
        kv = analytic_kv_bytes(mgr, len(ids))
    else:
        L = model.config.num_hidden_layers
        Hkv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        d = model.config.hidden_size // model.config.num_attention_heads
        fp16 = 2
        kv_tok = Hkv * d * fp16 * 2
        dense_full = L * len(ids) * kv_tok
        kv = {
            "per_block_bytes": 0,
            "lowrank_block_bytes": 0,
            "residual_block_bytes_max": 0,
            "store_alloc_bytes": dense_full,
            "store_used_bytes": dense_full,
            "dense_full_bytes": dense_full,
            "num_blocks_layer0": 0,
            "dense_len_layer0": len(ids),
            "res_tokens_layer0": 0,
            "ratio_used_vs_dense": 1.0,
        }

    res = {
        "prompt_tokens": len(ids), "gen_tokens": len(gen_ids),
        "prefill_s": prefill_s, "decode_s": decode_s,
        "decode_tps": len(gen_ids) / decode_s if decode_s > 0 else None,
        "mx_peak_gb": mx_peak_prefill,
        "mx_decode_peak_gb": _get_peak_gb(),
        "mx_active_end_gb": _get_active_gb(),
        "needle_found": NEEDLE in text,
        "output_preview": text[:200],
        "kv": kv,
    }

    if is_compressed:
        try: w.close()
        except Exception: pass
        del w, model, mgr; gc.collect()
    else:
        del model; gc.collect()

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", type=int, required=True)
    ap.add_argument("--modes", nargs="+", default=["compressed", "exact"])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", help="run ONE cell: mode,ctx (used by subprocess driver)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()

    os.chdir(ACTIVE)
    from bench_common import build_niah_prompt, _load_ref_tokenizer  # noqa
    tok = _load_ref_tokenizer()

    if args.single:
        mode, ctx = args.single.split(","); ctx = int(ctx)
        os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if mode == "compressed" else "0"
        text, _ = build_niah_prompt(ctx, tok)
        r = run_cell(ctx, args.gen, text, args.model)
        r.update({"mode": mode, "ctx": ctx})
        print("__CELL__ " + json.dumps(r))
        return

    model_id = args.model
    print(f"\n=== Running benchmarks with model: {model_id} ===", flush=True)

    # driver: spawn one subprocess per cell for clean memory isolation
    import subprocess
    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = []
    for ctx in args.ctx:
        for mode in args.modes:
            print(f">>> {mode} ctx={ctx} (Model: {model_id})", flush=True)
            env = os.environ.copy()
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--single", f"{mode},{ctx}", "--gen", str(args.gen),
                   "--ctx", str(ctx), "--out", out_path, "--model", model_id]
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            line = [l for l in p.stdout.splitlines() if l.startswith("__CELL__")]
            if line:
                r = json.loads(line[-1][len("__CELL__ "):])
                results.append(r)
                print(f"    prefill={r['prefill_s']:.1f}s tps={r['decode_tps']:.1f} "
                      f"needle={r['needle_found']} mx_peak={r['mx_peak_gb']:.2f} "
                      f"mx_decode={r['mx_decode_peak_gb']:.2f} "
                      f"store={r['kv']['store_used_bytes']/1e9:.2f}GB "
                      f"dense={r['kv']['dense_full_bytes']/1e9:.2f}GB", flush=True)
            else:
                print("    FAILED:\n" + p.stdout[-800:] + "\n" + p.stderr[-800:], flush=True)
                results.append({"mode": mode, "ctx": ctx, "status": "error",
                                "stderr": p.stderr[-2000:]})
            with open(out_path, "w") as f:
                json.dump({"results": results}, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
