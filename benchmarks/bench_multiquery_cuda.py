#!/usr/bin/env python3
"""Serving N queries over ONE context — the prefix-reuse experiment.

WHAT THIS MEASURES, AND WHY IT IS THE DIFFERENTIATOR
----------------------------------------------------
DKV compresses a context ONCE. Attention-observation eviction cannot: SnapKV
and H2O rank prefix tokens by attention from an observation window, and that
window is the tail of the prompt — i.e. THE QUERY. Change the query and the
selection changes, so the eviction has to be redone, and redoing it means
materialising the full KV again. That is the same cost that caps their context
ceiling (FINDINGS_LOG 1.2).

Under prefix caching or multi-turn serving a context is compressed once and
reused across many queries, so this is where that difference shows up as
throughput rather than as an argument. The paper currently asserts it; this
measures it.

HOW EACH ARM IS SERVED — each gets the best strategy IT can legitimately use:

  dense    prefill the context once, then per query append the query tokens,
           generate, and crop the cache back. Reuse is genuine.
  dkv      one session; the wrapper's O(1) prefix check makes prefill
           incremental over the resident cache (hf_dkv_wrapper.py:1680).
  snapkv   CANNOT reuse. Its selection depends on the query, so context+query
           must be prefilled and evicted from scratch every time. That is not
           a handicap imposed here, it is what the method requires.

Reported as total wall time for N queries and, more usefully, the MARGINAL
cost of query 2..N — which is what a serving system actually pays.

Correctness is checked, not assumed: every arm must answer the same N queries,
and the outputs are recorded so a run that served nothing cannot look fast.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL                            # noqa: E402
from code_fingerprint import decode_fingerprint                  # noqa: E402

QUESTIONS = [
    "What is the main topic of the text above?",
    "Name one specific detail mentioned in the text.",
    "What is the overall tone of the passage?",
    "Summarise the passage in one sentence.",
    "What subject area does this text belong to?",
    "Mention a term that appears in the text.",
    "Is the text technical or narrative?",
    "What would be a suitable title for this text?",
    "Identify one entity referred to in the text.",
    "State one claim made in the text.",
    "What kind of document does this appear to be?",
    "Give one keyword from the passage.",
    "Does the passage describe a process or an event?",
    "What is discussed near the beginning of the text?",
    "Name a concept the text refers to.",
    "In one word, what is the passage about?",
]


def run_point(args) -> Dict[str, Any]:
    import torch
    from context_ladder import build_filler
    import kv_baselines as KB

    res: Dict[str, Any] = {"arm": args.arm, "ctx": args.ctx,
                           "n_queries": args.n, "pattern": args.pattern}

    if args.arm == "dkv":
        os.environ.setdefault("DKV_RSVD_SEED", "1234")
        os.environ.setdefault("DKV_SVD_SEED", "1234")
        cwd = os.getcwd()
        os.chdir(ACTIVE)
        sys.path.insert(0, ACTIVE)
        from serving.decode_config import apply_best_decode_defaults
        apply_best_decode_defaults()
        from serving.hf_dkv_wrapper import DKVHFWrapper
        w = DKVHFWrapper(model_id=args.model,
                         config={"preset": args.preset,
                                 "quantization": args.quant or None})
        w.ensure_loaded()
        tok, model = w.tokenizer, None
        os.chdir(cwd)
    else:
        from run_longbench_cuda import load_plain
        tok, model = load_plain(args.model, args.quant, KB.needs_eager(args.arm))
        w = None

    ctx_text = build_filler(tok, args.ctx)
    qs = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.n)]
    res["ctx_actual"] = len(tok(ctx_text, add_special_tokens=False).input_ids)

    torch.cuda.reset_peak_memory_stats()
    per_query, texts, reuse_hits = [], [], []

    if args.arm == "dkv":
        # ONE session. WHICH PATTERN IS SERVED MATTERS, and the first version
        # of this harness measured only the one DKV cannot do.
        #
        # The prefix check (hf_dkv_wrapper.py:1758) reuses only when the new
        # prompt STRICTLY EXTENDS the stored session, because the session
        # records prompt+completion:
        #
        #   append       ctx q1 a1 q2  extends  ctx q1 a1  -> reuse fires
        #   independent  ctx q2        is SHORTER          -> guard fails,
        #                                                     session cleared
        #
        # So 'independent' measures an unsupported pattern, and it cannot show
        # a difference anyway -- SnapKV cannot reuse there either. 'append' is
        # the multi-turn pattern where DKV appends while SnapKV must re-evict
        # against its new observation window: the only place the claim is
        # actually testable.
        sid = f"mq-{args.ctx}-{time.time_ns()}"
        w.active_session = sid
        transcript = ctx_text
        for q in qs:
            prompt = transcript + "\n\nQuestion: " + q + "\nAnswer:"
            torch.cuda.synchronize()
            _cap = io.StringIO()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(_cap):
                out = w.generate(prompt, max_new_tokens=args.gen,
                                 temperature=0.0, top_p=1.0,
                                 repetition_penalty=1.0)
            torch.cuda.synchronize()
            per_query.append(time.perf_counter() - t0)
            _txt = _cap.getvalue()
            sys.stdout.write(_txt)
            reuse_hits.append(int('Reusing KV cache' in _txt))
            texts.append((out or "")[-120:])
            if args.pattern == "append":
                # generate() returns prompt+completion, so this IS the
                # extended transcript the next turn must strictly extend.
                transcript = out or prompt
        try:
            w.clear_session(sid)
        except Exception:                                        # noqa: BLE001
            pass

    elif args.arm == "dense":
        # Prefill the context ONCE, then per query append/crop. This is the
        # prefix-cache pattern a real server uses.
        ctx_ids = tok(ctx_text, add_special_tokens=False).input_ids
        past, _ = KB.chunked_prefill(model, ctx_ids, "cuda", args.chunk)
        base_len = len(ctx_ids)
        for q in qs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            qi = tok("\n\nQuestion: " + q + "\nAnswer:",
                     add_special_tokens=False).input_ids
            pos = torch.tensor([list(range(base_len, base_len + len(qi)))], device="cuda")
            with torch.no_grad():
                o = model(input_ids=torch.tensor([qi], device="cuda"),
                          position_ids=pos, past_key_values=past, use_cache=True)
                past = o.past_key_values
                lg = o.logits[0, -1]
                cur, gen = base_len + len(qi), []
                for _ in range(args.gen):
                    nid = int(torch.argmax(lg))
                    gen.append(nid)
                    o = model(input_ids=torch.tensor([[nid]], device="cuda"),
                              position_ids=torch.tensor([[cur]], device="cuda"),
                              past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    lg = o.logits[0, -1]
                    cur += 1
            if args.pattern == "independent":
                # Restore the shared prefix. Needs an EXACT rollback, which a
                # linear-attention layer's fixed-size recurrent state cannot
                # give -- on hybrid models this raises, and that is a property
                # of the architecture, not a defect in this harness.
                past.crop(base_len)
            else:
                # append: the conversation legitimately grows, nothing to undo.
                base_len = cur
            torch.cuda.synchronize()
            per_query.append(time.perf_counter() - t0)
            texts.append(tok.decode(gen, skip_special_tokens=True)[:120])

    else:
        # Eviction: selection depends on the query, so the whole context must be
        # prefilled and evicted again for every one. Not a handicap imposed
        # here -- it is what the method requires.
        bparams = json.loads(args.baseline_params)
        transcript = ctx_text
        for q in qs:
            prompt = transcript + "\n\nQuestion: " + q + "\nAnswer:"
            ids = tok(prompt, add_special_tokens=False).input_ids
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = KB.run_baseline(model, tok, ids, args.arm, "cuda", args.gen,
                                set(), args.chunk, bparams)
            torch.cuda.synchronize()
            per_query.append(time.perf_counter() - t0)
            txt = (r.get("text") or "")
            texts.append(txt[:120])
            if args.pattern == "append":
                transcript = prompt + txt

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    marginal = per_query[1:] or per_query
    res.update({
        "total_s": sum(per_query),
        "first_query_s": per_query[0],
        # What a serving system actually pays once the context is warm.
        "marginal_s": statistics.median(marginal),
        "per_query_s": [round(x, 3) for x in per_query],
        "peak_gb": peak, "vram_total_gb": total,
        "spilled": bool(peak > total * 0.94),
        "n_answered": sum(1 for t in texts if t.strip()),
        # How many queries actually hit the prefix cache. 0 with a flat
        # timing curve means the check never matched; >0 with a flat
        # curve would mean it matched and bought nothing -- a completely
        # different finding.
        "reuse_hits": sum(reuse_hits) if reuse_hits else None,
        "reuse_of": len(reuse_hits) or None,
        "sample_texts": texts[:3],
    })
    return res


def report(paths: List[str]) -> None:
    rows = []
    for p in paths:
        st = ResumableJSONL(p, config=None, strict_config=False, read_only=True)
        rows += [r for r in st.load_latest().values() if not r.get("error")]
        st.close()
    if not rows:
        print("no multiquery results")
        return
    ctxs = sorted({r["ctx"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    for title, key in [("first query (s) — includes compressing the context", "first_query_s"),
                       ("MARGINAL per query (s) — what a warm server pays", "marginal_s"),
                       ("total for N queries (s)", "total_s"),
                       ("peak VRAM (GB)", "peak_gb")]:
        print(f"\n=== {title} ===")
        print(f"{'arm':>10} " + " ".join(f"{c:>10}" for c in ctxs))
        print("-" * (11 + 11 * len(ctxs)))
        for a in arms:
            cells = []
            for c in ctxs:
                m = [r for r in rows if r["arm"] == a and r["ctx"] == c]
                v = m[0].get(key) if m else None
                mark = "*" if (m and m[0].get("spilled")) else " "
                cells.append(f"{v:>9.2f}{mark}" if v is not None else f"{'-':>10}")
            print(f"{a:>10} " + " ".join(cells))
    bad = [r for r in rows if r.get("n_answered", 0) < r.get("n_queries", 0)]
    if bad:
        print(f"\n[warn] {len(bad)} point(s) produced fewer answers than queries "
              f"— a fast run that served nothing is not a fast run.")
    print("\n* = spilled to host memory; that timing is bandwidth, not compute.")


def main():
    from msvc_env import ensure_msvc
    ensure_msvc()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--arms", nargs="+", default=["dense", "dkv", "snapkv"])
    ap.add_argument("--preset", default="mid", choices=["low", "mid", "high", "ultra"])
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--baseline-params", default='{"budget": 2016, "window": 32}')
    ap.add_argument("--pattern", default="append",
                    choices=["append", "independent"],
                    help="append: multi-turn, each prompt extends the last "
                         "-- the pattern the prefix check supports, and where "
                         "SnapKV must re-evict against a new observation "
                         "window. independent: one context, a different query "
                         "each time, which needs an exact rollback DKV has no "
                         "API for and hybrid models cannot do at all.")
    ap.add_argument("--out", default="")
    ap.add_argument("--arm", default="")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--point-json", default="")
    ap.add_argument("--report", nargs="*", default=None)
    args = ap.parse_args()

    if args.report is not None:
        paths = []
        for pat in (args.report or ["paper/results/multiquery/*.jsonl"]):
            paths += sorted(glob.glob(pat))
        if not paths:
            raise SystemExit("no multiquery files matched")
        return report(paths)

    if args.point_json:
        r = run_point(args)
        with open(args.point_json, "w", encoding="utf-8") as f:
            json.dump(r, f)
        return

    import subprocess
    import tempfile
    tag = args.model.split("/")[-1]
    # The run key is arm@ctx, so an append run would collide with the
    # independent rows already in the old file and be skipped on resume.
    # The pattern belongs in the filename, not only in the row.
    out = args.out or os.path.join(REPO, "paper", "results", "multiquery",
                                   f"{tag}_{args.preset}_n{args.n}_"
                                   f"{args.pattern}.jsonl")
    cfg = {"model": args.model, "preset": args.preset, "quant": args.quant,
           "n_queries": args.n, "gen": args.gen, "chunk": args.chunk,
           "dkv_decode_rev": decode_fingerprint(), "protocol": "multiquery-v1"}
    store = ResumableJSONL(out, config=cfg)
    done = store.load_done()
    tmp = tempfile.mkdtemp(prefix="dkv-mq-")
    for arm in args.arms:
        for ctx in sorted(args.contexts):
            key = f"{arm}@{ctx}"
            if key in done:
                print(f"  skip {key}")
                continue
            pj = os.path.join(tmp, f"{arm}_{ctx}.json")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--model", args.model, "--arm", arm, "--ctx", str(ctx),
                   "--preset", args.preset, "--quant", args.quant,
                   "--n", str(args.n), "--gen", str(args.gen),
                   "--chunk", str(args.chunk),
                   "--baseline-params", args.baseline_params,
                   # MUST be forwarded. The parent used args.pattern for the
                   # FILENAME while the child fell back to its default, so a run
                   # launched as --pattern independent wrote append data into a
                   # file named _independent. Caught by reuse_hits: 7 of 8 cache
                   # hits in an "independent" file is impossible, because that
                   # pattern can never reuse.
                   "--pattern", args.pattern,
                   "--point-json", pj]
            print(f"  running {key} ...", flush=True)
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as f:
                    r = json.load(f)
            else:
                r = {"arm": arm, "ctx": ctx, "error": (p.stderr or "")[-300:]}
            store.append(key, **r)
            print(f"    {key}: first={r.get('first_query_s','-')} "
                  f"marginal={r.get('marginal_s','-')} "
                  f"answered={r.get('n_answered','-')}/{args.n}", flush=True)
    store.close()
    report([out])


if __name__ == "__main__":
    main()
