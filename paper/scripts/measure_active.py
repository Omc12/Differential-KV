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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")
NEEDLE = "OMEGA-7741-DELTA"


def _mx_peak(mx):
    for o, n in ((mx, "get_peak_memory"), (getattr(mx, "metal", None), "get_peak_memory")):
        if o is not None and hasattr(o, n):
            try: return float(getattr(o, n)()) / 1e9
            except Exception: pass
    return None


def _mx_reset_peak(mx):
    for o, n in ((mx, "reset_peak_memory"), (getattr(mx, "metal", None), "reset_peak_memory")):
        if o is not None and hasattr(o, n):
            try: getattr(o, n)(); return
            except Exception: pass


def _mx_active(mx):
    for o, n in ((mx, "get_active_memory"), (getattr(mx, "metal", None), "get_active_memory")):
        if o is not None and hasattr(o, n):
            try: return float(getattr(o, n)()) / 1e9
            except Exception: pass
    return None


def analytic_kv_bytes(mgr, seq_len):
    """DiffKV store footprint (fixed pre-allocated pool + dense) vs full-KV, in bytes.
    Pools are pre-allocated at max_blocks, so the store footprint is BOUNDED &
    context-independent; we report both the allocated (real) and used footprint."""
    L = mgr.num_layers; Hkv = mgr.kv_heads; d = mgr.head_dim
    B = mgr.block_size; r = mgr.rank; M = mgr.max_blocks; Dmax = mgr.max_dense_len
    per_block = ((B - 1) * r * 2          # U fp16
                 + 2 * Hkv * r * d * 2     # V_K,V_V fp16
                 + 2 * Hkv * d * 2         # anchors fp16
                 + 8)                      # scale fp32 + seq_len int32
    dense_alloc = Dmax * Hkv * d * 2 * 2   # K,V fp16 over full buffer
    s0 = mgr.sessions.get("bench")
    nb = s0["num_blocks"][0] if s0 else 0
    dl = s0["dense_lens"][0] if s0 else 0
    store_alloc = L * (M * per_block + dense_alloc)
    store_used = L * (nb * per_block + dl * Hkv * d * 2 * 2)
    dense_full = L * seq_len * Hkv * d * 2 * 2
    return {
        "per_block_bytes": per_block,
        "store_alloc_bytes": store_alloc,
        "store_used_bytes": store_used,
        "dense_full_bytes": dense_full,
        "num_blocks_layer0": int(nb),
        "dense_len_layer0": int(dl),
        "ratio_used_vs_dense": (dense_full / store_used) if store_used else None,
    }


def run_cell(ctx, gen, prompt_text):
    import numpy as np, torch
    import mlx.core as mx
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper

    _mx_reset_peak(mx)
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    w = DiffKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=cfg)
    w.ensure_loaded()
    tok, mgr, model = w.tokenizer, w.manager, w.model
    ids = tok.encode(prompt_text)

    # warmup (compile kernels)
    try:
        mgr.clear_session("warm"); w._session_token_ids["warm"] = []
        mgr.init_session("warm", prefill_len=1)
        mgr.register_prefill_tokens("warm", torch.tensor([ids[0]], dtype=torch.long))
        model._diffkv_session_ids = ["warm"]
        _ = model(torch.tensor([[ids[0]]]), torch.tensor([[0]])).logits[0, -1].cpu().numpy()
        mgr.clear_session("warm")
    except Exception:
        pass

    sid = "bench"; mgr.clear_session(sid); w._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    model._diffkv_session_ids = [sid]

    CH = 512; out = None
    t0 = time.perf_counter()
    for cs in range(0, len(ids), CH):
        ch = ids[cs:cs + CH]
        out = model(torch.tensor([ch]), torch.tensor([list(range(cs, cs + len(ch)))]))
        mgr.compress_deferred_prefill_blocks(sid)
    logits = out.logits[0, -1].cpu().numpy()
    prefill_s = time.perf_counter() - t0

    mx_peak_prefill = _mx_peak(mx)
    # ── isolate decode-phase memory: reset peak at the boundary ──
    mx.eval(); _mx_reset_peak(mx)

    cur = len(ids); gen_ids = []
    t0 = time.perf_counter()
    for _ in range(gen):
        nid = int(np.argmax(logits)); gen_ids.append(nid)
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        out = model(torch.tensor([[nid]]), torch.tensor([[cur]]))
        logits = out.logits[0, -1].cpu().numpy(); cur += 1
    decode_s = time.perf_counter() - t0

    text = tok.decode(gen_ids)
    kv = analytic_kv_bytes(mgr, len(ids))
    res = {
        "prompt_tokens": len(ids), "gen_tokens": len(gen_ids),
        "prefill_s": prefill_s, "decode_s": decode_s,
        "decode_tps": len(gen_ids) / decode_s if decode_s > 0 else None,
        "mx_peak_gb": mx_peak_prefill,
        "mx_decode_peak_gb": _mx_peak(mx),
        "mx_active_end_gb": _mx_active(mx),
        "needle_found": NEEDLE in text,
        "output_preview": text[:200],
        "kv": kv,
    }
    try: w.close()
    except Exception: pass
    del w, model, mgr; gc.collect(); mx.clear_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", type=int, required=True)
    ap.add_argument("--modes", nargs="+", default=["compressed", "exact"])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", help="run ONE cell: mode,ctx (used by subprocess driver)")
    args = ap.parse_args()

    sys.path.insert(0, ACTIVE); sys.path.insert(0, BENCH)
    os.chdir(ACTIVE)
    from bench_common import build_niah_prompt, _load_ref_tokenizer  # noqa
    tok = _load_ref_tokenizer()

    if args.single:
        mode, ctx = args.single.split(","); ctx = int(ctx)
        os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if mode == "compressed" else "0"
        text, _ = build_niah_prompt(ctx, tok)
        r = run_cell(ctx, args.gen, text)
        r.update({"mode": mode, "ctx": ctx})
        print("__CELL__ " + json.dumps(r))
        return

    # driver: spawn one subprocess per cell for clean memory isolation
    import subprocess
    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = []
    for ctx in args.ctx:
        for mode in args.modes:
            print(f">>> {mode} ctx={ctx}", flush=True)
            env = os.environ.copy()
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--single", f"{mode},{ctx}", "--gen", str(args.gen),
                   "--ctx", str(ctx), "--out", out_path]
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
