#!/usr/bin/env python3
"""Prefill-time + recall harness for DKV_SPARSE_PREFILL (HANDOFF §DSA).

niah_recall.py measures DECODE tps; sparse prefill's win is PREFILL wall-clock, which this
harness times directly. It reuses the SAME hard on-topic NIAH prompt (bench_common) and the
same chunked-prefill + greedy-decode loop as niah_recall, then reports prefill seconds, decode
tps, and needle recall per ctx. Toggle the engine with the env flag and run it twice:

    # dense prefill (baseline)
    DKV_COMPRESSED_DECODE=1 python benchmarks/sparse_prefill_bench.py --ctx 8192 16384 32768
    # sparse prefill
    DKV_COMPRESSED_DECODE=1 DKV_SPARSE_PREFILL=1 python benchmarks/sparse_prefill_bench.py --ctx 8192 16384 32768

Prefill time is the authoritative number here; decode tps is approximate (kernel recompiles).
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

NEEDLE = "OMEGA-7741-DELTA"


def run(model_id, contexts, gen, rank=16):
    import numpy as np
    import torch
    import mlx.core as mx  # noqa: F401
    sys.path.insert(0, HERE)
    from bench_common import build_niah_prompt   # the harder, on-topic NIAH prompt
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper

    quant_val = None if (model_id.startswith(".") or model_id.startswith("/")) else "int4"
    cfg = {"quantization": quant_val, "rank": rank, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DKVHFWrapper(model_id=model_id, config=cfg)
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model

    dec = os.environ.get("DKV_COMPRESSED_DECODE", "auto")
    # Report the ACTUAL knob values the manager resolved (env override OR ON-defaults), not the
    # env string — otherwise an unset env misleadingly prints the harness's own default.
    print(f"SPARSE_PREFILL={getattr(mgr,'_sparse_prefill',False)}  COMPRESSED_DECODE={dec}  gen={gen}  "
          f"KMIN={getattr(mgr,'_sp_kmin','?')} FRAC={getattr(mgr,'_sp_frac','?')} "
          f"WIN={getattr(mgr,'_sp_window','?')} MIN={getattr(mgr,'_sp_min_ctx','?')}", flush=True)
    print(f"{'ctx':>7} {'ntok':>6} {'prefill_s':>9} {'fwd_s':>7} {'comp_s':>7} {'pf_tok/s':>9} {'dec_tps':>7} {'recall':>6}   sample",
          flush=True)

    results = []
    for ctx in contexts:
        prompt, _ = build_niah_prompt(ctx, tok)
        ids = tok.encode(prompt)
        sid = "spb"
        mgr.clear_session(sid)
        if not hasattr(wrapper, "_session_token_ids"):
            wrapper._session_token_ids = {}
        wrapper._session_token_ids[sid] = []
        mgr.init_session(sid, prefill_len=len(ids))
        mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
        model._dkv_session_ids = [sid]

        CH = 512
        output = None
        comp_s = 0.0
        # time the whole chunked prefill (forward + streaming compression), as in niah_recall
        mx.eval()
        t_pf = time.perf_counter()
        for cs in range(0, len(ids), CH):
            chunk = ids[cs:cs + CH]
            ct = torch.tensor([chunk], dtype=torch.long)
            pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
            output = model(ct, pt)
            _tc = time.perf_counter()
            mgr.compress_deferred_prefill_blocks(sid)
            comp_s += time.perf_counter() - _tc
        logits = output.logits[0, -1].cpu().numpy()
        prefill_s = time.perf_counter() - t_pf
        fwd_s = prefill_s - comp_s

        cur = len(ids)
        generated = []
        nid = int(np.argmax(logits))
        generated.append(nid)
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        output = model(torch.tensor([[nid]], dtype=torch.long),
                       torch.tensor([[cur]], dtype=torch.long))
        logits = output.logits[0, -1].cpu().numpy()
        cur += 1

        t0 = time.perf_counter()
        steps = 0
        for _ in range(gen - 1):
            nid = int(np.argmax(logits))
            generated.append(nid)
            steps += 1
            if NEEDLE in tok.decode(generated):
                break
            mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
            output = model(torch.tensor([[nid]], dtype=torch.long),
                           torch.tensor([[cur]], dtype=torch.long))
            logits = output.logits[0, -1].cpu().numpy()
            cur += 1
        dt = time.perf_counter() - t0

        text = tok.decode(generated)
        ok = NEEDLE in text
        dec_tps = steps / dt if dt > 0 else 0.0
        pf_toks = len(ids) / prefill_s if prefill_s > 0 else 0.0
        results.append((ctx, len(ids), prefill_s, ok))
        print(f"{ctx:>7} {len(ids):>6} {prefill_s:>9.2f} {fwd_s:>7.2f} {comp_s:>7.2f} {pf_toks:>9.0f} {dec_tps:>7.1f} "
              f"{('Y' if ok else 'N'):>6}   {text[:80]!r}", flush=True)

    print(f"\nRECALL: {sum(1 for _,_,_,ok in results if ok)}/{len(results)} cells", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192, 16384, 32768])
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    args = ap.parse_args()
    run(args.model, args.ctx, args.gen, rank=args.rank)
