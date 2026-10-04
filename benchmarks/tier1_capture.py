#!/usr/bin/env python3
"""Capture real K, V and queries from a dense run, for offline store screening.

For each prompt: chunked dense prefill, then greedy decode of --gen tokens.
Hooks on q_proj / k_proj / v_proj of the chosen layers record the PRE-RoPE
projections (DKV stores keys unrotated and rotates on read), so the offline
screen can apply RoPE itself at the true positions:

    k, v      [T_prompt + gen, H_kv * D]   every position, fp16
    q_dec     [gen, H_q * D]               the queries decode actually issues
    q_pre     [~T/16, H_q * D]             a subsample of prefill queries, for
                                           query statistics (importance weights)
    q_pre_pos positions of q_pre

USAGE
    python benchmarks/tier1_capture.py --model ibm-granite/granite-4.2-8b \\
        --layers 1 8 16 24 32 39 --out paper/results/diag/tier1
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 8, 16, 24, 32, 39])
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--lb", nargs="*", default=["qasper:15-15", "multifieldqa_en:3-3",
                                                "hotpotqa:1-1"])
    ap.add_argument("--ruler", nargs="*", default=["16384/niah_multikey_2:0-0",
                                                   "16384/niah_multikey_3:0-0",
                                                   "16384/qa_1:0-0"])
    ap.add_argument("--qsub", type=int, default=16, help="keep every Nth prefill query")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "paper",
                                                  "results", "diag", "tier1"))
    args = ap.parse_args()

    import torch
    from run_longbench_cuda import load_plain, derive_stop_ids
    from decode_fidelity import build_items
    tok, model = load_plain(args.model, args.quant, False)
    stop = derive_stop_ids(tok)
    items = build_items(tok, args.model, args.lb, args.ruler, args.max_length)
    # Decoder layers and the rotary module, found by ROLE (any HF decoder,
    # including ones nested in a multimodal wrapper): the longest ModuleList
    # named "layers", and the first module whose class ends in RotaryEmbedding.
    layers = max((m for n, m in model.named_modules()
                  if n.endswith("layers") and isinstance(m, torch.nn.ModuleList)),
                 key=len)
    rotary = next(m for _n, m in model.named_modules()
                  if type(m).__name__.endswith("RotaryEmbedding"))
    os.makedirs(args.out, exist_ok=True)

    buf = {}
    phase = {"dec": False, "pos0": 0}

    def mk(li, name):
        def hook(_m, _i, out):
            x = out[0].detach().to(torch.float16).cpu()
            x = x.reshape(x.shape[0], -1)                        # [L, H*D]
            d = buf.setdefault(li, {"k": [], "v": [], "q_pre": [], "q_pre_pos": [],
                                    "q_dec": []})
            if name in ("k", "v"):
                d[name].append(x)
            elif phase["dec"]:
                d["q_dec"].append(x)
            else:
                p0 = phase["pos0"]
                idx = [i for i in range(x.shape[0]) if (p0 + i) % args.qsub == 0]
                if idx:
                    d["q_pre"].append(x[idx])
                    d["q_pre_pos"].extend(p0 + i for i in idx)
        return hook

    def src(at, n):
        # The tensor attention actually rotates: after a per-head norm when the
        # architecture has one (q_norm / k_norm), else the projection output.
        for name in ((n + "_norm",) if n in ("q", "k") else ()) + (n + "_proj",):
            m = getattr(at, name, None)
            if m is not None:
                return m
        raise SystemExit("attention module %s has no %s_proj (fused QKV is not "
                         "supported by this capture yet)" % (type(at).__name__, n))

    hs = []
    for li in args.layers:
        at = layers[li].self_attn
        hs += [src(at, "q").register_forward_hook(mk(li, "q")),
               src(at, "k").register_forward_hook(mk(li, "k")),
               at.v_proj.register_forward_hook(mk(li, "v"))]
    try:
        for key, prompt in items:
            buf.clear()
            ids = tok(prompt).input_ids
            from transformers import DynamicCache
            cache = DynamicCache(config=model.config)
            phase["dec"] = False
            with torch.no_grad():
                for i in range(0, len(ids), 1024):
                    ch = ids[i:i + 1024]
                    phase["pos0"] = i
                    out = model(input_ids=torch.tensor([ch], device="cuda"),
                                position_ids=torch.tensor([list(range(i, i + len(ch)))],
                                                          device="cuda"),
                                past_key_values=cache, use_cache=True)
                logits = out.logits[0, -1]
                phase["dec"] = True
                gen, cur = [], len(ids)
                for _ in range(args.gen):
                    t = int(logits.argmax())
                    gen.append(t)
                    if t in stop:
                        break
                    out = model(input_ids=torch.tensor([[t]], device="cuda"),
                                position_ids=torch.tensor([[cur]], device="cuda"),
                                past_key_values=cache, use_cache=True)
                    logits = out.logits[0, -1]
                    cur += 1
            # RoPE tables and the attention scale, read off the LIVE model so the
            # screen needs no per-architecture code.
            P = len(ids) + len(gen)
            dummy = torch.zeros(1, 1, 1, device="cuda", dtype=torch.float32)
            pos = torch.arange(P, device="cuda")[None]
            try:
                cos, sin = rotary(dummy, pos)
            except Exception:                                    # noqa: BLE001
                # multi-axis RoPE (text positions on every axis)
                cos, sin = rotary(dummy, pos.unsqueeze(0).expand(3, -1, -1))
            at0 = layers[args.layers[0]].self_attn
            hd = getattr(at0, "head_dim", None) or cos.shape[-1]
            rec = {"key": key, "T": len(ids), "gen": gen, "model": args.model,
                   "cos": cos[0].float().cpu(), "sin": sin[0].float().cpu(),
                   "scale": float(getattr(at0, "scaling", hd ** -0.5)),
                   "head_dim": int(hd), "layers": {}}
            for li, d in buf.items():
                rec["layers"][li] = {
                    "k": torch.cat(d["k"]), "v": torch.cat(d["v"]),
                    "q_pre": torch.cat(d["q_pre"]),
                    "q_pre_pos": torch.tensor(d["q_pre_pos"]),
                    "q_dec": torch.cat(d["q_dec"]) if d["q_dec"] else None}
            name = key.replace("/", "_").replace("#", "_")
            torch.save(rec, os.path.join(args.out, name + ".pt"))
            print("captured %s: T=%d gen=%d layers=%s" % (key, len(ids), len(gen),
                                                         sorted(rec["layers"])), flush=True)
            del cache
            torch.cuda.empty_cache()
    finally:
        for h in hs:
            h.remove()


if __name__ == "__main__":
    main()
