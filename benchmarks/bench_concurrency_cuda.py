#!/usr/bin/env python3
"""How many long-context sessions can each arm keep resident on one card?

A single request's context ceiling is one systems question; the one a
deployment asks is how many users fit. This holds sessions RESIDENT at once --
each prefilled on its own distinct prompt of `--ctx` tokens and decoded a few
tokens, then kept alive -- and adds sessions until the card spills. The answer
per arm is the largest count that stayed clean.

WHAT THIS IS AND IS NOT
-----------------------
Sessions are prefilled one after another and each decodes at batch size 1; this
measures RESIDENCY (memory held per live session plus the transient of the one
prefill in flight), not batched throughput. Eviction arms pay their prefill
transient once at a time and then hold only their budget, so they are expected
to fit many sessions; dense holds everything. That is the trade-off being
measured, not a flaw of the setup.

Spill is detected exactly as the ladder does it: allocation or reservation
above 94% of the card, or a session prefill more than 2.5x slower than the
first one (WDDM pages to host memory instead of raising).

DKV's pool ceiling defaults to half the free VRAM at load. That is a context
cap, not a memory reservation (the pool is lazy), and it would stop DKV
admitting sessions long before memory does, so it is set explicitly here via
DKV_POOL_BUDGET_GB and recorded.

    python benchmarks/bench_concurrency_cuda.py
    python benchmarks/bench_concurrency_cuda.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)

SPILL_FRACTION = 0.94
CLIFF_RATIO = 2.5
DEFAULT_ARMS = ["dense", "dkv", "snapkv", "streamingllm_chunked", "h2o_chunked"]


def distinct_ids(base, ctx, i):
    """Session i's prompt: the filler rotated, so no two sessions share a prefix."""
    off = (i * 997) % max(1, len(base) - ctx)
    return base[off:off + ctx]


def run_arm(args):
    import torch
    rec = {"arm": args.arm, "model": args.model, "ctx": args.ctx,
           "n_max": args.n_max, "sessions": []}
    try:
        if args.arm == "dkv":
            os.environ["DKV_POOL_BUDGET_GB"] = str(args.dkv_pool_gb)
            os.environ.setdefault("DKV_RSVD_SEED", "1234")
            os.environ.setdefault("DKV_SVD_SEED", "1234")
            os.chdir(ACTIVE)
            sys.path.insert(0, ACTIVE)
            from serving.hf_dkv_wrapper import DKVHFWrapper
            w = DKVHFWrapper(model_id=args.model,
                             config={"preset": "mid", "quantization": "nf4"})
            w.ensure_loaded()
            tok = w.tokenizer
            rec["dkv_pool_budget_gb"] = args.dkv_pool_gb
        else:
            from run_longbench_cuda import load_plain
            import kv_baselines as KB
            tok, model = load_plain(args.model, "nf4", False)
        from context_ladder import build_filler
        base = tok(build_filler(tok, args.ctx + 8192),
                   add_special_tokens=False).input_ids
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        rec["vram_total_gb"] = total
        rec["weights_gb"] = torch.cuda.memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        held = []
        first_s = None
        status = "ok"
        for i in range(args.n_max):
            ids = distinct_ids(base, args.ctx, i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.arm == "dkv":
                w.active_session = f"conc-{i}"
                w.generate(tok.decode(ids), max_new_tokens=args.gen,
                           temperature=0.0, top_p=1.0, repetition_penalty=1.0)
            else:
                if args.arm == "dense":
                    past, logits = KB.chunked_prefill(model, ids, "cuda", 1024)
                elif args.arm == "snapkv":
                    past, logits, _ = KB._evict_by_observed_attention(
                        model, ids, "cuda", 1024, budget=2016, window=32,
                        pool_kernel=7, recency_window=0)
                elif args.arm == "streamingllm_chunked":
                    past, logits, _, _ = KB._streamingllm_chunked(
                        model, ids, "cuda", 1024, 4, 2044)
                elif args.arm == "h2o_chunked":
                    past, logits, _, _ = KB._h2o_chunked(
                        model, ids, "cuda", 256, 2048, 512)
                else:
                    raise ValueError(args.arm)
                cur = len(ids)
                with torch.no_grad():
                    for _ in range(args.gen):
                        nid = int(torch.argmax(logits).item())
                        out = model(input_ids=torch.tensor([[nid]], device="cuda"),
                                    position_ids=torch.tensor([[cur]], device="cuda"),
                                    past_key_values=past, use_cache=True)
                        past, logits = out.past_key_values, out.logits[0, -1].float()
                        cur += 1
                del out
                held.append(past)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            first_s = first_s or dt
            s = {"i": i, "wall_s": round(dt, 2),
                 "resident_gb": torch.cuda.memory_allocated() / 1e9,
                 "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                 "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
            rec["sessions"].append(s)
            print(json.dumps(s), flush=True)
            if max(s["peak_gb"], s["peak_reserved_gb"]) > total * SPILL_FRACTION:
                status = "spilled"
            elif i > 0 and dt > first_s * CLIFF_RATIO:
                status = "degraded"
            if status != "ok":
                s["status"] = status
                break
        clean = [s for s in rec["sessions"] if "status" not in s]
        rec["max_clean_sessions"] = len(clean)
        rec["hit_n_max"] = status == "ok"
        rec["status"] = status
    except Exception as e:                                       # noqa: BLE001
        msg = str(e)
        rec["status"] = ("oom" if "out of memory" in msg.lower() else "error")
        rec["error"] = f"{type(e).__name__}: {msg[:400]}"
        rec["max_clean_sessions"] = len(rec["sessions"])
    with open(args.point_json, "w", encoding="utf-8") as f:
        json.dump(rec, f)


def report(path):
    from checkpoint import ResumableJSONL
    s = ResumableJSONL(path, read_only=True)
    for k, r in sorted(s.load_latest().items()):
        ss = r.get("sessions") or []
        per = ((ss[-1]["resident_gb"] - r.get("weights_gb", 0)) / len(ss)) if ss else 0
        cap = f">={r['max_clean_sessions']}" if r.get("hit_n_max") else str(r.get("max_clean_sessions"))
        print(f"{r.get('arm'):>22}  ctx {r.get('ctx')}  sessions clean {cap:>5}  "
              f"~{per:.3f} GB resident/session  status {r.get('status')}"
              + (f"  ({r.get('error')})" if r.get("error") else ""))


def main():
    from msvc_env import ensure_msvc
    ensure_msvc()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--dkv-pool-gb", type=float, default=9.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arm", default="")
    ap.add_argument("--point-json", default="")
    args = ap.parse_args()
    out = args.out or os.path.join(REPO, "paper", "results", "concurrency",
                                   f"{args.model.split('/')[-1]}_ctx{args.ctx}.jsonl")
    if args.report:
        return report(out)
    if args.point_json:
        return run_arm(args)

    from checkpoint import ResumableJSONL
    store = ResumableJSONL(out, config={"model": args.model, "ctx": args.ctx,
                                        "n_max": args.n_max, "gen": args.gen,
                                        "dkv_pool_gb": args.dkv_pool_gb})
    done = store.load_done()
    tmp = tempfile.mkdtemp(prefix="dkv-conc-")
    for arm in args.arms:
        if arm in done:
            print(f"skip {arm}")
            continue
        pj = os.path.join(tmp, f"{arm}.json")
        print(f"run {arm}", flush=True)
        t0 = time.time()
        try:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--arm", arm,
                            "--model", args.model, "--ctx", str(args.ctx),
                            "--n-max", str(args.n_max), "--gen", str(args.gen),
                            "--dkv-pool-gb", str(args.dkv_pool_gb),
                            "--point-json", pj], cwd=REPO, timeout=7200)
        except subprocess.TimeoutExpired:
            pass
        if os.path.exists(pj):
            with open(pj, encoding="utf-8") as f:
                rec = json.load(f)
        else:
            rec = {"arm": arm, "status": "died", "error": "child wrote nothing"}
        rec["wall_s"] = round(time.time() - t0, 1)
        store.append(arm, **rec)
        print(f"  {arm}: {rec.get('status')} clean={rec.get('max_clean_sessions')}",
              flush=True)
    store.close()
    report(out)


if __name__ == "__main__":
    main()
