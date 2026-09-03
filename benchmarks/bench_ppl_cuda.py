#!/usr/bin/env python3
"""Long-context perplexity, measured so that it actually exercises compression.

WHY THE OBVIOUS IMPLEMENTATION MEASURES NOTHING
-----------------------------------------------
Perplexity is teacher-forced, so the naive version runs one forward over the
whole document and averages the token NLLs. On this system that is a PREFILL
measurement, and DKV's prefill is exact -- logit fidelity reads KL 0.00000
against a true dense control at the first token. Such a harness would report a
difference of ~0 and prove nothing about compression.

So the document is processed in CHUNKS, and the NLL of each chunk is scored
while attending the COMPRESSED history of the chunks before it. Early chunks
(before DKV engages at ~4,970 tokens, see FINDINGS_LOG 4.4) are excluded from
the average, because until then DKV is bit-identical to dense by construction
and including them dilutes the measurement toward zero.

Both arms are scored over exactly the same token positions.

Data is WikiText-103 (already in the HF cache) or PG19, concatenated to the
requested length. Reported as mean NLL and perplexity over the scored window,
plus the delta against dense.

USAGE
    python benchmarks/bench_ppl_cuda.py --model Qwen/Qwen3.5-4B \
        --arms dense dkv --contexts 16384 32768
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL                            # noqa: E402
from code_fingerprint import decode_fingerprint                  # noqa: E402

# Below this, DKV has not engaged and is bit-identical to dense; scoring those
# positions would dilute every delta toward zero. Measured: 3,965 tokens -> 0
# compressed blocks, 4,970 -> 140.
SKIP_TOKENS = 5120


def load_text(name: str, min_chars: int):
    """Return (text, source). Tries several spellings before giving up.

    Every point failed on the first run with
        ValueError: Couldn't find cache for wikitext for config
        'wikitext-103-raw-v1'. Available configs in the cache:
        ['wikitext-2-raw-v1']
    -- the bare repo id `wikitext` is no longer resolvable on the Hub (it moved
    to Salesforce/wikitext) so `datasets` fell back to the local cache, which
    holds only the -2 config. Rather than pin one spelling, try the candidates
    in order and RECORD WHICH ONE WAS USED, so a perplexity number can always
    be traced to the corpus it came from.
    """
    from datasets import load_dataset
    if name == "pg19":
        cands = [(("deepmind/pg19",), {"split": "test", "streaming": True}),
                 (("pg19",), {"split": "test", "streaming": True})]
    else:
        cands = [(("Salesforce/wikitext", "wikitext-103-raw-v1"), {"split": "test"}),
                 (("wikitext", "wikitext-103-raw-v1"), {"split": "test"}),
                 (("Salesforce/wikitext", "wikitext-2-raw-v1"), {"split": "test"}),
                 (("wikitext", "wikitext-2-raw-v1"), {"split": "test"})]
    last = None
    for a, kw in cands:
        try:
            ds = load_dataset(*a, **kw)
            src = "/".join(str(x) for x in a)
            buf, n = [], 0
            for row in ds:
                t = row.get("text") or ""
                if not t.strip():
                    continue
                buf.append(t)
                n += len(t)
                if n >= min_chars:
                    break
            if n < min_chars // 2:
                last = f"{src}: only {n} chars, needed {min_chars}"
                continue
            return "\n\n".join(buf), src
        except Exception as e:                                   # noqa: BLE001
            last = f"{a}: {type(e).__name__}: {str(e)[:120]}"
    raise SystemExit(f"no usable corpus for '{name}'. Last: {last}")


def score(model_or_wrapper, tok, ids, chunk, is_dkv, sid_prefix):
    """Mean NLL per token over positions >= SKIP_TOKENS, chunk by chunk.

    The lm_head output is captured directly, so this works for the DKV wrapper
    (whose generate() would otherwise only expose text) and for a plain model
    through the identical code path -- the two arms cannot drift apart in how
    they are scored.
    """
    import torch
    import torch.nn.functional as F

    model = model_or_wrapper.model if is_dkv else model_or_wrapper
    total_nll, total_tok = 0.0, 0
    past = None
    sid = f"{sid_prefix}-{os.getpid()}"
    if is_dkv:
        model_or_wrapper.active_session = sid

    with torch.inference_mode():
        for cs in range(0, len(ids) - 1, chunk):
            ch = ids[cs:cs + chunk]
            if len(ch) < 2:
                break
            inp = torch.tensor([ch], device="cuda")
            pos = torch.tensor([list(range(cs, cs + len(ch)))], device="cuda")
            out = model(input_ids=inp, position_ids=pos,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            if cs + len(ch) <= SKIP_TOKENS:
                continue                      # DKV not engaged yet
            logits = out.logits[0, :-1].float()
            tgt = torch.tensor(ch[1:], device="cuda")
            keep = max(0, SKIP_TOKENS - cs)   # partial chunk at the boundary
            if keep:
                logits, tgt = logits[keep:], tgt[keep:]
            if tgt.numel() == 0:
                continue
            nll = F.cross_entropy(logits, tgt, reduction="sum")
            total_nll += float(nll)
            total_tok += int(tgt.numel())
            del out, logits
    if is_dkv:
        try:
            model_or_wrapper.clear_session(sid)
        except Exception:                                        # noqa: BLE001
            pass
    mean = total_nll / max(1, total_tok)
    return {"nll": mean, "ppl": math.exp(min(20.0, mean)), "scored_tokens": total_tok}


def run_point(args) -> Dict[str, Any]:
    import torch
    res: Dict[str, Any] = {"arm": args.arm, "ctx": args.ctx}
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
        tok = w.tokenizer
        os.chdir(cwd)
        obj, is_dkv = w, True
    else:
        from run_longbench_cuda import load_plain
        tok, model = load_plain(args.model, args.quant, eager=False)
        obj, is_dkv = model, False

    text, res["corpus"] = load_text(args.data, args.ctx * 8)
    ids = tok(text, add_special_tokens=False).input_ids[:args.ctx]
    res["ctx_actual"] = len(ids)
    torch.cuda.reset_peak_memory_stats()
    res.update(score(obj, tok, ids, args.chunk, is_dkv, f"ppl{args.ctx}"))
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return res


def report(paths: List[str]) -> None:
    rows = []
    for p in paths:
        st = ResumableJSONL(p, config=None, strict_config=False, read_only=True)
        rows += [r for r in st.load_latest().values() if not r.get("error")]
        st.close()
    if not rows:
        print("no ppl results")
        return
    ctxs = sorted({r["ctx"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    print(f"\n=== perplexity (scored only past {SKIP_TOKENS} tokens, "
          f"where DKV is actually engaged) ===")
    print(f"{'arm':>10} " + " ".join(f"{c:>12}" for c in ctxs))
    print("-" * (11 + 13 * len(ctxs)))
    base = {}
    for a in arms:
        cells = []
        for c in ctxs:
            m = [r for r in rows if r["arm"] == a and r["ctx"] == c]
            if not m:
                cells.append(f"{'-':>12}")
                continue
            ppl = m[0]["ppl"]
            if a == "dense":
                base[c] = ppl
            d = f" ({ppl - base[c]:+.3f})" if (a != "dense" and c in base) else ""
            cells.append(f"{ppl:>7.3f}{d:>5}")
        print(f"{a:>10} " + " ".join(cells))
    print("\nDeltas are against dense at the same context. Positive = worse.")


def main():
    from msvc_env import ensure_msvc
    ensure_msvc()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--arms", nargs="+", default=["dense", "dkv"])
    ap.add_argument("--preset", default="mid", choices=["low", "mid", "high", "ultra"])
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--data", default="wikitext", choices=["wikitext", "pg19"])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="")
    ap.add_argument("--arm", default="")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--point-json", default="")
    ap.add_argument("--report", nargs="*", default=None)
    args = ap.parse_args()

    if args.report is not None:
        paths = []
        for pat in (args.report or ["paper/results/ppl/*.jsonl"]):
            paths += sorted(glob.glob(pat))
        if not paths:
            raise SystemExit("no ppl files matched")
        return report(paths)

    if args.point_json:
        r = run_point(args)
        with open(args.point_json, "w", encoding="utf-8") as f:
            json.dump(r, f)
        return

    import subprocess
    import tempfile
    tag = args.model.split("/")[-1]
    out = args.out or os.path.join(REPO, "paper", "results", "ppl",
                                   f"{tag}_{args.data}_{args.preset}.jsonl")
    cfg = {"model": args.model, "preset": args.preset, "quant": args.quant,
           "data": args.data, "chunk": args.chunk, "skip_tokens": SKIP_TOKENS,
           "dkv_decode_rev": decode_fingerprint(), "protocol": "ppl-chunked-v1"}
    store = ResumableJSONL(out, config=cfg)
    done = store.load_done()
    tmp = tempfile.mkdtemp(prefix="dkv-ppl-")
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
                   "--data", args.data, "--chunk", str(args.chunk),
                   "--point-json", pj]
            print(f"  running {key} ...", flush=True)
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as f:
                    r = json.load(f)
            else:
                r = {"arm": arm, "ctx": ctx, "error": (p.stderr or "")[-300:]}
            store.append(key, **r)
            print(f"    {key}: ppl={r.get('ppl','-')} "
                  f"tokens={r.get('scored_tokens','-')}", flush=True)
    store.close()
    report([out])


if __name__ == "__main__":
    main()
