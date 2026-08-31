#!/usr/bin/env python3
"""Does the residual storage format cost long-context recall on CUDA? (int4/int8/fp16)

WHY A NEW HARNESS AND NOT THE NEEDLE SWEEP
------------------------------------------
The single-needle sweep CANNOT answer this question and must not be used for it.
Measured on MLX over 3 contexts x 3 depths it scored int4 8/9 and fp16 7/9 -- it
ranked the CATASTROPHIC format above the good one, because one needle in a short
prompt is recoverable from almost any surviving representation. CUDA's standing
"needle sweep 9/9" was taken that way and is therefore not evidence about
residual format at all.

What discriminates is FOUR verbatim codes that must come back exactly, at
randomised depths, over enough trials to put an interval on the result. On MLX
that instrument read int4 0/48, int8 41/48, fp16 42/48 at ctx=20000.

WHY THE FORMAT SHOULD MATTER AT ALL (the mechanism this is testing)
------------------------------------------------------------------
Residuals are the EXACT-COPY tokens. `DKV_RESIDUAL_EXCLUDE_SVD=1` deliberately
drops a residual token's lossy SVD twin *because* the residual is meant to be the
faithful copy. Quantising residuals coarsely therefore leaves those tokens with
no accurate representation ANYWHERE in the store -- unlike quantising the basis,
where the residual path still holds an exact copy. That is architectural, not
MLX-specific, which is why it has to be measured here too.

INSTRUMENT CHECKS THIS HARNESS RUNS ON ITSELF (read these before any score)
--------------------------------------------------------------------------
1. ALLOCATOR FINGERPRINT per arm, read off the LIVE pool object after the first
   session allocates: residual_quant, residual_quant_bits, and the actual
   comp_res_k_q shape/nbytes. An arm whose allocator is identical to another
   arm's did not perturb anything and is REPORTED AS SUCH rather than scored.
   This is not paranoia: until e38f3cd1 "int8" allocated a 4-bit packed_width and
   was a byte-identical alias for int4, and nothing outside the allocator could
   tell. Two arms with the same fingerprint are one arm run twice.
2. PAIRED BY CONSTRUCTION. Prompts are built ONCE, before any model is loaded,
   and reused across arms; the sha1 of each prompt is recorded per (ctx, trial)
   so the pairing is checkable in the saved rows. Arms differ in the residual
   format and nothing else.
3. A/A NOISE FLOOR. `fp16_aa` is a second, independently constructed fp16 arm.
   Greedy decode makes it a tie by construction, so if it does NOT tie `fp16`
   exactly, decode is not deterministic across wrapper builds and no other
   difference in the table means anything.
4. NEEDLE TOKENISATION. Every code is built from a name checked to split into
   whole words and digit groups only, on THIS model, before any trial runs.
   Partial-word needles (rigorous_eval's OMEGA-7741-DELTA splits ' O'|'ME'|'GA')
   make recall a coin flip at a measured top-2 margin of 0.19 logits, which would
   swamp the effect under test.

SCORING is exact string containment of all four codes, scored per needle
(4 needles x N trials per cell), reported with a Wilson 95% interval. Both a
STRICT match and a case/punctuation-normalised match are reported: they should
agree, and a gap between them is a case-flip finding, not a retrieval one.

    python colab/residual_format_niah_cuda.py
    python colab/residual_format_niah_cuda.py --ctx 16384 --trials 12
    python colab/residual_format_niah_cuda.py --arms fp16,int8 --ctx 32768
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

# Whole-word-tokenising names; each is re-verified against the live tokenizer at
# startup and any that fragments on THIS model is dropped before trials start.
NAME_POOL = [
    "Falcon", "Titan", "Everest", "Cobra", "Phoenix", "Jaguar", "Viking",
    "Meteor", "Harbor", "Lantern", "Marble", "Cedar", "Anchor", "Bishop",
    "Compass", "Dolphin", "Ember", "Glacier", "Hammer", "Ivory",
]
ORDINALS = ["first", "second", "third", "fourth"]
N_NEEDLES = 4

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)
QUESTION = ("What are the four secret passcodes? List them all in order, "
            "separated by commas.")

# Arm -> the residual_quant value passed in the wrapper CONFIG DICT (which beats
# the environment in DKVConfig._get_str, so an inherited DKV_RESIDUAL_QUANT
# cannot silently redefine an arm).
ARMS = {
    "fp16":    "none",   # unquantised residual buffers -- the reference
    "fp16_aa": "none",   # A/A noise floor; must tie `fp16` exactly
    "int8":    "int8",   # the shipped default since 2026-08-31
    "int4":    "int4",   # the format measured at 0/48 on MLX -- positive control
}


def _norm(s):
    return "".join(c for c in s.upper() if c.isalnum())


def wilson(k, n, z=1.96):
    """95% Wilson score interval, in percent. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100.0 * max(0.0, (c - r) / d), 100.0 * min(1.0, (c + r) / d))


def tokenises_cleanly(tok, code):
    """True if `code` splits into whole words, digit groups and separators only."""
    parts = [tok.decode([i]) for i in
             tok(" " + code, add_special_tokens=False).input_ids]
    words = {w.lower() for w in code.replace("-", " ").split()}
    for p in parts:
        s = p.strip()
        if not s or s.isdigit() or s in ("-", "_"):
            continue
        if s.lower() not in words:
            return False
    return True


def build_trials(tok, ctxs, trials, seed, names):
    """All prompts for every (ctx, trial), built ONCE and shared by every arm."""
    filler_ids = tok(FILLER, add_special_tokens=False).input_ids
    out = {}
    for ctx in ctxs:
        body_ids = (filler_ids * (ctx // max(1, len(filler_ids)) + 2))[:ctx]
        for t in range(trials):
            rng = random.Random(seed + 1000 * t + ctx)
            codes = [n + "-" + str(rng.randint(1000, 9999)) + "-" + str(rng.randint(1000, 9999))
                     for n in rng.sample(names, N_NEEDLES)]
            # Randomised depths, kept apart so two needles never land in one block.
            while True:
                depths = sorted(rng.uniform(0.05, 0.95) for _ in range(N_NEEDLES))
                if all(b - a >= 0.10 for a, b in zip(depths, depths[1:])):
                    break
            marks = [int(len(body_ids) * d) for d in depths]
            parts, prev = [], 0
            for m, ordn, code in zip(marks, ORDINALS, codes):
                parts.append(tok.decode(body_ids[prev:m]))
                parts.append(" The " + ordn + " secret passcode is " + code + ". ")
                prev = m
            parts.append(tok.decode(body_ids[prev:]))
            body = "".join(parts) + "\n\n" + QUESTION
            prompt = tok.apply_chat_template([{"role": "user", "content": body}],
                                             tokenize=False, add_generation_prompt=True)
            out[(ctx, t)] = {
                "prompt": prompt,
                "codes": codes,
                "depths": [round(d, 4) for d in depths],
                "sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12],
                "prompt_tokens": len(tok(prompt).input_ids),
            }
    return out


def pool_fingerprint(w):
    """Read the LIVE allocator, not the config that was asked for."""
    try:
        p = w.manager.native_pool
    except Exception:                                              # noqa: BLE001
        return {"error": "no native_pool"}
    q = getattr(p, "comp_res_k_q", None)
    return {
        "residual_quant": getattr(p, "residual_quant", None),
        "residual_quant_bits": getattr(p, "residual_quant_bits", None),
        "max_residual_tokens": getattr(p, "max_residual_tokens", None),
        "comp_res_k_q_shape": (None if q is None else tuple(q.shape)),
        "comp_res_k_q_dtype": (None if q is None else str(q.dtype)),
        "comp_res_k_q_bytes": (0 if q is None else int(q.nbytes)),
    }


def run_arm(arm, fmt, model_id, cases, ctxs, trials, max_new):
    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    print("\n### arm " + arm + " (residual_quant=" + repr(fmt) + ") -- loading "
          + model_id, flush=True)
    w = PyTorchDKVHFWrapper(model_id=model_id,
                            config={"mode": "fp16", "residual_quant": fmt},
                            device="cuda")
    w.ensure_loaded()
    rows, fp = [], None
    for ctx in ctxs:
        for t in range(trials):
            case = cases[(ctx, t)]
            sid = arm + "_" + str(ctx) + "_" + str(t)
            w.clear_session(sid)
            if hasattr(w, "_session_token_ids"):
                w._session_token_ids[sid] = []
            w.active_session = sid
            t0 = time.perf_counter()
            out = w.generate(prompt=case["prompt"], max_new_tokens=max_new,
                             temperature=0.0, top_p=1.0, repetition_penalty=1.0)
            dt = time.perf_counter() - t0
            out = out.rsplit("assistant", 1)[-1].strip()
            if fp is None:                       # after the first real allocation
                fp = pool_fingerprint(w)
                print("  allocator: " + str(fp), flush=True)
            strict = [c in out for c in case["codes"]]
            loose = [_norm(c) in _norm(out) for c in case["codes"]]
            rows.append({"arm": arm, "ctx": ctx, "trial": t, "sha1": case["sha1"],
                         "prompt_tokens": case["prompt_tokens"],
                         "depths": case["depths"], "codes": case["codes"],
                         "strict": strict, "loose": loose,
                         "n_strict": sum(strict), "n_loose": sum(loose),
                         "sec": round(dt, 2), "out": out[:240]})
            print("  %-8s ctx=%6d trial=%2d strict=%d/%d loose=%d/%d (%.1fs) %r"
                  % (arm, ctx, t, sum(strict), N_NEEDLES, sum(loose), N_NEEDLES,
                     dt, out[:70]), flush=True)
            w.clear_session(sid)
    try:
        w.close()
    except Exception:                                              # noqa: BLE001
        pass
    del w
    gc.collect()
    torch.cuda.empty_cache()
    return rows, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", default="16384,32768")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--arms", default="fp16,fp16_aa,int8,int4")
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", default=os.path.join(_ROOT, "benchmarks", "results",
                                                  "residual_format_niah_cuda.json"))
    a = ap.parse_args()
    ctxs = [int(c) for c in a.ctx.split(",")]
    arms = [s.strip() for s in a.arms.split(",")]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit("unknown arm " + repr(arm) + "; known: " + str(sorted(ARMS)))

    # The format is passed per arm in the CONFIG DICT. Clearing these makes sure
    # nothing inherited from the shell can redefine an arm behind its back, and
    # that the bit width is derived from the format name rather than pinned.
    for k in ("DKV_RESIDUAL_QUANT", "DKV_RESIDUAL_QUANT_BITS"):
        if k in os.environ:
            print("[harness] unsetting inherited " + k + "=" + repr(os.environ[k]))
            del os.environ[k]
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    names = [n for n in NAME_POOL if tokenises_cleanly(tok, n + "-1234-5678")]
    print("[harness] %d/%d names tokenise cleanly on %s: %s"
          % (len(names), len(NAME_POOL), a.model, names), flush=True)
    if len(names) < N_NEEDLES + 2:
        raise SystemExit("too few unambiguous names for this tokenizer")

    cases = build_trials(tok, ctxs, a.trials, a.seed, names)
    for ctx in ctxs:
        toks = [cases[(ctx, t)]["prompt_tokens"] for t in range(a.trials)]
        print("[harness] ctx=%d: prompt tokens %d-%d, %d trials x %d needles = "
              "%d needles/arm" % (ctx, min(toks), max(toks), a.trials, N_NEEDLES,
                                  a.trials * N_NEEDLES), flush=True)

    all_rows, fps = [], {}
    for arm in arms:
        rows, fps[arm] = run_arm(arm, ARMS[arm], a.model, cases, ctxs,
                                 a.trials, a.max_new)
        all_rows += rows

    # -- instrument checks, printed BEFORE the scores -----------------------
    print("\n=== ALLOCATOR FINGERPRINTS (live pool, per arm) ===", flush=True)
    for arm in arms:
        print("  %-8s %s" % (arm, fps[arm]), flush=True)
    distinct = {}
    for arm in arms:
        key = json.dumps(fps[arm], sort_keys=True, default=str)
        distinct.setdefault(key, []).append(arm)
    inert = [v for v in distinct.values()
             if len(v) > 1 and set(v) != {"fp16", "fp16_aa"}]
    if inert:
        print("  !! ARMS WITH AN IDENTICAL ALLOCATOR: " + str(inert) + ". These are "
              "the SAME arm run twice; their scores are not a comparison.", flush=True)
    else:
        print("  ok: every scored format has a distinct allocator "
              "(fp16/fp16_aa share one by design).", flush=True)

    if "fp16" in arms and "fp16_aa" in arms:
        for ctx in ctxs:
            aa = [r for r in all_rows if r["arm"] == "fp16_aa" and r["ctx"] == ctx]
            bb = [r for r in all_rows if r["arm"] == "fp16" and r["ctx"] == ctx]
            same = all(x["strict"] == y["strict"] for x, y in zip(aa, bb))
            print("  A/A ctx=%d: fp16 vs fp16_aa identical per needle: %s%s"
                  % (ctx, same, "" if same else "   <- DECODE IS NOT DETERMINISTIC; "
                     "no other row in the table means anything"), flush=True)

    print("\n=== RECALL (needle-level, Wilson 95%) ===", flush=True)
    print("  %-9s %7s %10s %7s %16s   %10s"
          % ("arm", "ctx", "strict", "%", "95% CI", "loose"), flush=True)
    summary = []
    for arm in arms:
        for ctx in ctxs:
            rs = [r for r in all_rows if r["arm"] == arm and r["ctx"] == ctx]
            n = len(rs) * N_NEEDLES
            ks = sum(r["n_strict"] for r in rs)
            kl = sum(r["n_loose"] for r in rs)
            lo, hi = wilson(ks, n)
            pct = 100.0 * ks / n if n else 0.0
            summary.append({"arm": arm, "ctx": ctx, "k_strict": ks, "k_loose": kl,
                            "n": n, "pct": pct, "ci95": [round(lo, 1), round(hi, 1)]})
            print("  %-9s %7d %6d/%-3d %6.1f%% [%5.1f, %5.1f]   %6d/%-3d"
                  % (arm, ctx, ks, n, pct, lo, hi, kl, n), flush=True)

    # -- PAIRED comparison against fp16 ------------------------------------
    # Unpaired Wilson intervals are the wrong test for this design and will
    # understate it: every arm sees the SAME prompts, so the informative
    # quantity is the DISCORDANT needles -- the ones one arm got and the other
    # did not. Two arms can share a rate and disagree on half their needles, or
    # differ by six points with every disagreement pointing the same way. Sign
    # test (exact binomial, two-sided) on the discordant pairs, which is McNemar
    # without the chi-square approximation that 6 events do not support.
    if "fp16" in arms:
        print("\n=== PAIRED vs fp16 (same prompts; discordant needles) ===", flush=True)
        for arm in arms:
            if arm == "fp16":
                continue
            for ctx in ctxs:
                ref = sorted([r for r in all_rows if r["arm"] == "fp16" and r["ctx"] == ctx],
                             key=lambda r: r["trial"])
                cur = sorted([r for r in all_rows if r["arm"] == arm and r["ctx"] == ctx],
                             key=lambda r: r["trial"])
                paired = all(x["sha1"] == y["sha1"] for x, y in zip(ref, cur))
                lost = sum(1 for x, y in zip(ref, cur)
                           for p, q in zip(x["strict"], y["strict"]) if p and not q)
                won = sum(1 for x, y in zip(ref, cur)
                          for p, q in zip(x["strict"], y["strict"]) if q and not p)
                nd = lost + won
                # Two-sided exact binomial at p=0.5 on the discordant pairs.
                pval = 1.0 if nd == 0 else min(1.0, 2.0 * sum(
                    math.comb(nd, i) for i in range(0, min(lost, won) + 1)) / (2.0 ** nd))
                print("  %-9s ctx=%6d  discordant=%2d (fp16 kept/%s lost %d, "
                      "%s kept/fp16 lost %d)  sign-test p=%.3f%s"
                      % (arm, ctx, nd, arm, lost, arm, won, pval,
                         "" if paired else "   <- PROMPTS NOT PAIRED"), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"model": a.model, "trials": a.trials, "seed": a.seed,
                   "fingerprints": fps, "summary": summary, "rows": all_rows},
                  f, indent=2, default=str)
    print("\nsaved " + a.out, flush=True)


if __name__ == "__main__":
    main()
