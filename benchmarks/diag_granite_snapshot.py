#!/usr/bin/env python3
"""What grows with context on granite that DKV does not compress?

The ladder says DKV stores 0.653 GB of KV on granite-4.2-8b at 16k against
dense's 2.684 GB, yet the two arms' peak-memory slopes are the same and both
stop at the same rung. So the peak is set by something other than the KV
store. diag_granite_residency.py tried to find it by walking gc.get_objects()
and hung on an 8B model; this uses the allocator's own record instead.

METHOD
------
torch.cuda.memory._record_memory_history() logs every allocation and free
during one prefill, with the Python stack that made it. Replaying that log
gives the exact set of blocks alive at the moment of peak allocation. Those
blocks are bucketed by the innermost non-torch frame that allocated them
(file:line function), so "what is alive at the peak" comes back as named
code sites with byte counts.

Run at two context lengths, the per-bucket growth per 1k tokens is the
answer: whatever bucket carries granite's slope is the uncompressed term.
Qwen3.5-4B runs as the control, where DKV does move the ceiling.

One subprocess per point (fresh allocator), resumable per point.

    python benchmarks/diag_granite_snapshot.py
    python benchmarks/diag_granite_snapshot.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)

OUT = os.path.join(REPO, "paper", "results", "diag", "granite_peak_snapshot.jsonl")
POINTS = [
    ("ibm-granite/granite-4.2-8b", "dkv", 8192),
    ("ibm-granite/granite-4.2-8b", "dkv", 16384),
    ("ibm-granite/granite-4.2-8b", "dense", 8192),
    ("ibm-granite/granite-4.2-8b", "dense", 16384),
    ("Qwen/Qwen3.5-4B", "dkv", 16384),
    ("Qwen/Qwen3.5-4B", "dkv", 32768),
]
TOP = 25


def site_of(frames) -> str:
    """Innermost frame that is not torch internals: the code that asked."""
    for fr in frames or []:
        fn = (fr.get("filename") or "").replace("\\", "/")
        if not fn or "/torch/" in fn or fn.startswith("<") or "/contextlib" in fn:
            continue
        short = fn
        for anchor in ("ACTIVE_RUNTIME/", "benchmarks/", "site-packages/"):
            if anchor in fn:
                short = fn.split(anchor, 1)[1]
                break
        return f"{short}:{fr.get('line')} {fr.get('name')}"
    return "<torch internal>"


def live_at_peak(snapshot) -> dict:
    """Replay the allocator trace; return the live set at max allocation."""
    traces = snapshot.get("device_traces") or [[]]
    events = traces[0]
    live = {}                   # addr -> (size, site)
    cur = peak = 0
    peak_live = {}
    for ev in events:
        act = ev.get("action")
        addr = ev.get("addr")
        if act == "alloc":
            size = int(ev.get("size", 0))
            live[addr] = (size, site_of(ev.get("frames")))
            cur += size
            if cur > peak:
                peak = cur
                peak_live = dict(live)
        elif act in ("free_requested", "free_completed"):
            got = live.pop(addr, None)
            if got is not None:
                cur -= got[0]
    buckets = defaultdict(lambda: [0, 0])
    for size, site in peak_live.values():
        buckets[site][0] += size
        buckets[site][1] += 1
    return {"traced_peak_gb": peak / 1e9, "n_events": len(events),
            "buckets": {k: {"gb": v[0] / 1e9, "n": v[1]}
                        for k, v in sorted(buckets.items(), key=lambda kv: -kv[1][0])[:TOP]}}


def run_point(model_id, arm, ctx, out_json):
    import torch
    rec = {"model": model_id, "arm": arm, "ctx": ctx}
    try:
        if arm == "dkv":
            os.environ.setdefault("DKV_RSVD_SEED", "1234")
            os.environ.setdefault("DKV_SVD_SEED", "1234")
            os.chdir(ACTIVE)
            sys.path.insert(0, ACTIVE)
            from serving.hf_dkv_wrapper import DKVHFWrapper
            w = DKVHFWrapper(model_id=model_id,
                             config={"preset": "mid", "quantization": "nf4"})
            w.ensure_loaded()
            tok = w.tokenizer
        else:
            from run_longbench_cuda import load_plain
            tok, model = load_plain(model_id, "nf4", False)
        from context_ladder import build_filler
        prompt = build_filler(tok, ctx)
        ids = tok(prompt, add_special_tokens=False).input_ids
        rec["ctx_actual"] = len(ids)
        torch.cuda.synchronize()
        rec["baseline_gb"] = torch.cuda.memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(
            enabled="all", context="alloc", stacks="python", max_entries=2_000_000)
        if arm == "dkv":
            w.active_session = "diag"
            w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0)
        else:
            import kv_baselines as KB
            past, _ = KB.chunked_prefill(model, ids, "cuda", 1024)
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        rec["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rec["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        rec.update(live_at_peak(snap))
        # traced_peak + baseline should reproduce peak_gb; if it does not,
        # the ring buffer dropped events and the buckets are incomplete.
        rec["trace_complete"] = abs(rec["traced_peak_gb"] + rec["baseline_gb"]
                                    - rec["peak_gb"]) < 0.05
        rec["status"] = "ok"
    except Exception as e:                                       # noqa: BLE001
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f)


def report():
    from checkpoint import ResumableJSONL
    s = ResumableJSONL(OUT, read_only=True)
    recs = s.load_latest()
    by = {(r["model"], r["arm"], r["ctx"]): r for r in recs.values() if r.get("status") == "ok"}
    for (m, arm, _c) in sorted({(k[0], k[1], 0) for k in by}):
        pts = sorted((k[2], v) for k, v in by.items() if k[0] == m and k[1] == arm)
        print(f"\n=== {m.split('/')[-1]}  {arm} ===")
        for c, r in pts:
            print(f"  ctx {c}: peak {r['peak_gb']:.2f} GB  baseline {r['baseline_gb']:.2f}"
                  f"  traced {r['traced_peak_gb']:.2f}  complete={r['trace_complete']}")
        if len(pts) >= 2:
            (c0, a), (c1, b) = pts[0], pts[-1]
            dk = (c1 - c0) / 1000.0
            sites = set(a["buckets"]) | set(b["buckets"])
            rows = sorted(((b["buckets"].get(s, {}).get("gb", 0)
                            - a["buckets"].get(s, {}).get("gb", 0)) / dk, s) for s in sites)
            print(f"  growth per 1k tokens, largest first "
                  f"(total {(b['peak_gb'] - a['peak_gb']) / dk:.4f} GB/1k):")
            for g, s in reversed(rows[-12:]):
                print(f"    {g:+.4f} GB/1k  {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--point", nargs=3, default=None)
    ap.add_argument("--point-json", default="")
    ap.add_argument("--out", default="",
                    help="store path; use a new one when DKV_* env differs")
    ap.add_argument("--points", nargs="*", default=None,
                    help="model|arm|ctx entries, replacing the default set")
    args = ap.parse_args()
    global OUT, POINTS
    if args.out:
        OUT = args.out
    if args.points:
        POINTS = [(p.split("|")[0], p.split("|")[1], int(p.split("|")[2]))
                  for p in args.points]
    if args.report:
        return report()
    if args.point:
        m, arm, ctx = args.point
        return run_point(m, arm, int(ctx), args.point_json)

    from checkpoint import ResumableJSONL
    store = ResumableJSONL(OUT, config={"points": [list(p) for p in POINTS],
                                        "preset": "mid", "quant": "nf4",
                                        "dkv_env": {k: v for k, v in os.environ.items()
                                                    if k.startswith("DKV_")} or None})
    done = store.load_done()
    tmp = tempfile.mkdtemp(prefix="diag-snap-")
    for m, arm, ctx in POINTS:
        key = f"{m}|{arm}|{ctx}"
        if key in done:
            print(f"skip {key}")
            continue
        pj = os.path.join(tmp, "p.json")
        if os.path.exists(pj):
            os.remove(pj)
        print(f"run {key}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, os.path.abspath(__file__), "--point", m, arm,
                        str(ctx), "--point-json", pj], cwd=REPO, timeout=5400)
        if os.path.exists(pj):
            with open(pj, encoding="utf-8") as f:
                rec = json.load(f)
        else:
            rec = {"model": m, "arm": arm, "ctx": ctx, "status": "died",
                   "error": "child wrote nothing"}
        rec["wall_s"] = round(time.time() - t0, 1)
        store.append(key, **rec)
        print(f"  {rec.get('status')} peak={rec.get('peak_gb')} "
              f"complete={rec.get('trace_complete')}", flush=True)
    store.close()
    report()


if __name__ == "__main__":
    main()
