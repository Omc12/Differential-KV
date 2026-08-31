#!/usr/bin/env python3
"""Does DKV_RSVD_MAX_RPROJ=32 cost needle recall?  A POWERED, PAIRED decision.

WHY THE EXISTING EVIDENCE CANNOT DECIDE
---------------------------------------
DKV_RSVD_MAX_RPROJ=32 is the last fidelity-affecting knob in the DKV_FAST
bundle.  It is worth real time (Gram eigh ~2.1 s -> ~0.015 s: the batched
cuSOLVER cliff at 32x32), but it CAPS every block's stored rank, and the
content-aware rank boost it replaces existed specifically to give DIGIT blocks
extra rank.  test_niah.py asserts a 6-digit needle.  So the cap's risk lands
exactly where the shipped test looks -- and that test runs n=3 with ONE fixed
needle (847291).

Four problems with deciding from that:

  1. n=3 HAS NO POWER.  A Wilson 95% interval on 3/3 spans [0.44, 1.00] --
     width 0.56.  It cannot distinguish "no regression" from "a third of
     needles now fail".  Re-running it more times does not fix this; the
     INSTRUMENT has to change.
  2. UNPAIRED comparison throws away the variance that matters.  Whether a
     needle is recalled depends enormously on the prompt (depth, tokenisation
     of the digits).  Comparing arm means across DIFFERENT prompts buries the
     effect in prompt variance.  Run both arms on the SAME prompt and the
     comparison is within-pair.
  3. A t-test on means is the WRONG TEST.  Recall is a BINARY outcome.  The
     paired binary test is McNemar's, which looks only at DISCORDANT pairs
     (the prompts where the two arms disagree) -- concordant pairs carry no
     information about a difference.
  4. ONE FIXED NEEDLE cannot detect a digit-block effect.  847291 forever means
     one tokenisation, sampled repeatedly.  Each replicate here draws a FRESH
     6-digit code, so the estimate is over the population of digit needles
     rather than over one.

WHAT THIS REPORTS
    per arm : recall + Wilson 95% CI (Wilson, not normal-approx: at proportions
              near 1.0 the normal interval runs past 1 and is meaningless)
    paired  : McNemar exact two-sided p on the discordant pairs
    A cap that is safe shows: overlapping CIs, few discordant pairs, p high.
    A cap that costs recall shows: losses >> wins among discordants, low p.

VALIDATE THE HARNESS BEFORE BELIEVING IT
    python colab/rproj_cap_decision.py --selftest
runs the statistics against synthetic outcomes with KNOWN answers, including a
reproduction of the n=3 case, and needs no GPU.

NOTE ON CONTEXT: below ~8k DKV routing does not engage ("routing is not engaged
for this context"), nothing is compressed, and BOTH arms are the same
computation -- the instrument check below will say so.  Run at >=16k.

USAGE
    python colab/rproj_cap_decision.py --selftest
    python colab/rproj_cap_decision.py --n 32 --ctx 16384
    python colab/rproj_cap_decision.py --n 16 --ctx 16384 \
        --model Qwen/Qwen2.5-7B-Instruct --quant nf4     # fits a 7B on 12 GB
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")


# --------------------------- statistics ---------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Deliberately NOT the normal approximation p +- z*sqrt(p(1-p)/n): at p=1.0
    that has ZERO width, which would report a 3/3 run as perfectly certain --
    the exact failure this harness exists to avoid.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(wins, losses):
    """Two-sided exact McNemar on discordant counts.

    wins   = prompts the CAP got right and the baseline got wrong
    losses = prompts the BASELINE got right and the cap got wrong
    Concordant pairs are excluded BY DESIGN -- they carry no information about
    a difference, and including them (as a naive accuracy comparison does) is
    what dilutes a real effect into "no significant difference".
    """
    n = wins + losses
    if n == 0:
        return 1.0
    p = 2.0 * _binom_cdf(min(wins, losses), n, 0.5)
    return min(1.0, p)


def discordance_upper(k, n):
    """95% one-sided upper bound on the probability that the two arms DISAGREE.

    This is the statistic that survives a ceiling.  Per-arm recall CIs go wide
    (or pin at 1.0) whenever the task is easy, which says nothing about the
    cap; but the PAIRED disagreement rate is well defined either way.  With
    k=0 discordant pairs in n, the exact one-sided bound is 1 - 0.05**(1/n)
    (the "rule of three" family) -- so n=80 clean pairs bounds the cap's effect
    at ~3.7% of prompts, a real claim, where "32/32 vs 32/32" alone is not.
    """
    if n == 0:
        return 1.0
    if k == 0:
        return 1.0 - 0.05 ** (1.0 / n)
    return wilson(k, n)[1]


def report(rows, label_a="cap_off", label_b="cap_32"):
    n = len(rows)
    ka = sum(1 for r in rows if r["a"])
    kb = sum(1 for r in rows if r["b"])
    wins = sum(1 for r in rows if r["b"] and not r["a"])
    losses = sum(1 for r in rows if r["a"] and not r["b"])
    la, ua = wilson(ka, n)
    lb, ub = wilson(kb, n)
    p = mcnemar_exact(wins, losses)
    print("")
    print("  n = %d paired prompts, fresh 6-digit code each" % n)
    print("  %8s: %d/%d = %.3f   Wilson95 [%.3f, %.3f]  width %.3f"
          % (label_a, ka, n, (ka / n if n else 0), la, ua, ua - la))
    print("  %8s: %d/%d = %.3f   Wilson95 [%.3f, %.3f]  width %.3f"
          % (label_b, kb, n, (kb / n if n else 0), lb, ub, ub - lb))
    db = discordance_upper(wins + losses, n)
    print("  discordant: cap WON %d, cap LOST %d   McNemar exact p = %.4f"
          % (wins, losses, p))
    print("  disagreement rate: %d/%d, 95%% upper bound %.3f  <- survives a ceiling"
          % (wins + losses, n, db))
    if n and (ua - la) > 0.25:
        print("  [!] UNDERPOWERED: baseline CI wider than 0.25 -- this n cannot clear the cap.")
    if p < 0.05:
        print("  -> SIGNIFICANT difference. Do NOT promote the cap on this evidence.")
    else:
        print("  -> no significant difference at this n (see CI width before calling it safe).")
    return {"n": n, "ka": ka, "kb": kb, "wins": wins, "losses": losses, "p": p,
            "disc_upper": db}


# --------------------------- self-test ---------------------------
def selftest():
    print("SELFTEST -- statistics against known answers (no GPU)\n")
    ok = True

    print("Case A: identical arms, n=32, both 30/32 -- expect p=1.0, no difference")
    rows = [{"a": i >= 2, "b": i >= 2} for i in range(32)]
    r = report(rows)
    ok &= (r["p"] == 1.0 and r["wins"] == 0 and r["losses"] == 0)

    print("\nCase B: clear regression -- cap loses 8, wins 0 -- expect p=0.0078")
    rows = [{"a": True, "b": i >= 8} for i in range(32)]
    r = report(rows)
    ok &= (abs(r["p"] - 2 * 0.5 ** 8) < 1e-9 and r["p"] < 0.05)

    print("\nCase C: THE ORIGINAL n=3 SITUATION -- 3/3 vs 2/3 (a 33% observed drop)")
    rows = [{"a": True, "b": i >= 1} for i in range(3)]
    r = report(rows)
    ok &= (r["p"] == 1.0)
    print("  ^ a 33% observed drop reports p=1.00. That run could neither clear")
    print("    the cap NOR catch a real regression. The n was the problem.")

    print("\nWilson width vs n at perfect recall (why n=3 decides nothing):")
    for n in (3, 8, 16, 32, 64):
        lo, hi = wilson(n, n)
        print("    n=%3d  %d/%d  [%.3f, %.3f]  width %.3f" % (n, n, n, lo, hi, hi - lo))

    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# --------------------------- GPU arms ---------------------------
def run(n, ctx, depths, seed, model, quant):
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    os.environ.setdefault("DKV_ROUTER", "residual")
    os.environ.pop("DKV_SRL_THRESHOLD", None)
    os.environ.pop("DKV_TOPK_BLOCKS", None)
    from serving.hf_dkv_wrapper import DKVHFWrapper
    sys.path.insert(0, os.path.join(ACTIVE, "tests"))
    from test_niah import make_niah_prompt

    import torch

    def pool_sig(wrap):
        """Numeric fingerprint of the compressed factors. ADVISORY ONLY.

        WHY A NULL-INSTRUMENT GUARD IS NEEDED AT ALL.  With RANK_BOOST=off the
        per-block ranks are already 32, so the cap does NOT truncate stored
        rank -- it only narrows the randomized-SVD projection from
        rank+oversamples (37) to 32.  If that failed to reach compression, both
        arms would be the same computation and every needle would trivially
        agree; "identical recall" would mean "the knob did nothing", NOT "the
        cap is safe".  Those two must be told apart.

        WHY THIS FINGERPRINT DOES NOT DO IT.  Measured 2026-08-31: the pool's
        allocation lifecycle is not controlled from here.  Slots freed by
        clear_session are reclaimed lazily, during a LATER generate, so the
        per-generate delta picks up the previous replicate's deferred frees --
        it came out negative for the second arm in 31/31 pairs regardless of
        the knob.  Clearing between arms and clearing after the pair both fail,
        in opposite directions.  Absolute sums fare no better (the second arm's
        pool still holds the first arm's blocks).  Do not read this column as
        evidence either way.

        THE ACTUAL POSITIVE CONTROL is colab/bench_prefill_arms_peak.py's
        rproj_off / rproj_32 arms: the cap is reproducibly -13.4% forward and
        -73 MB peak at 32k (-13.2% / -37 MB at 16k), with peak identical across
        replicates within each arm.  A knob that never reached compression
        could not move either number.  The cap is live; the recall comparison
        below is therefore NOT vacuous.
        """
        pool = getattr(wrap.manager, "native_pool", None)
        if pool is None:
            return None
        vals = []
        for nm in ("U", "V_KV", "scales", "anchors_KV"):
            t = getattr(pool, nm, None)
            if torch.is_tensor(t) and t.numel():
                vals.append(round(float(t.detach().float().abs().sum().item()), 3))
        return tuple(vals)

    rng = random.Random(seed)
    _cfg = {"rank": 32}
    if quant:
        _cfg["quantization"] = quant
    w = DKVHFWrapper(model, config=_cfg)
    w.ensure_loaded()

    def recall_once(prompt, code, tag):
        # Report the signature DELTA, not the running total.  The pool
        # ACCUMULATES across sessions, so a cumulative sum grows on every
        # generate no matter what the knob does -- comparing totals would
        # report "the knob moved" for a knob doing nothing at all.  The delta
        # is what THIS session's compression actually contributed.
        before = pool_sig(w)
        w.active_session = tag
        resp = w.generate(prompt=prompt, max_new_tokens=16, temperature=0.0,
                          top_p=1.0, repetition_penalty=1.0)
        after = pool_sig(w)
        ptoks = len(w.tokenizer.encode(prompt))
        ids = w._session_token_ids.get(w.active_session or "default", [])
        gen = w.tokenizer.decode(ids[ptoks:], skip_special_tokens=True)
        tail = resp
        qi = resp.rfind("What is the special code")
        if qi >= 0:
            tail = resp[qi:]
        hit = (code in gen) or (code in tail)
        # NOTE: deliberately NO clear_session here.  Clearing between the two
        # arms makes their deltas incomparable -- the freed slots show up as a
        # NEGATIVE delta for whichever arm follows the clear, which looks like
        # "the knob moved the factors" when it is only the allocator moving.
        # Both arms of a pair run against the same monotonically growing pool
        # so their increments can be compared; the pair is cleared afterwards.
        d = None
        if before is not None and after is not None and len(before) == len(after):
            d = tuple(round(after[i] - before[i], 3) for i in range(len(after)))
        return hit, d

    rows = []
    for i in range(n):
        code = "%06d" % rng.randint(100000, 999999)
        depth = depths[i % len(depths)]
        needle = "The special code is %s." % code
        question = "What is the special code? Answer in exactly the 6-digit code number."
        prompt = make_niah_prompt(w.tokenizer, ctx, depth, needle, question)
        # BOTH arms on the SAME prompt (paired). The cap is read at call time in
        # _compress_layer_blocks_gpu_inner, so flipping it between generates is
        # sufficient -- no reload needed.
        os.environ["DKV_RSVD_MAX_RPROJ"] = "0"
        a, sig_a = recall_once(prompt, code, "capoff-%d" % i)

        os.environ["DKV_RSVD_MAX_RPROJ"] = "32"
        b, sig_b = recall_once(prompt, code, "cap32-%d" % i)

        # Clear BOTH sessions now that both deltas are recorded, so 2n generates
        # cannot exhaust the pool.
        for _t in ("capoff-%d" % i, "cap32-%d" % i):
            try:
                w.clear_session(_t)
            except Exception:
                pass
        moved = (sig_a is not None and sig_b is not None and sig_a != sig_b)
        rows.append({"a": a, "b": b, "code": code, "depth": depth,
                     "knob_moved": bool(moved),
                     "sig_a": list(sig_a) if sig_a else None,
                     "sig_b": list(sig_b) if sig_b else None})
        if not moved:
            print("        sig_off=%r sig_32=%r" % (sig_a, sig_b), flush=True)
        print("  [%3d/%d] code=%s depth=%.2f  cap_off=%s  cap_32=%s%s%s"
              % (i + 1, n, code, depth, "HIT " if a else "MISS",
                 "HIT " if b else "MISS",
                 "   <-- DISCORDANT" if a != b else "",
                 "" if moved else "   [!] knob did not move the factors"), flush=True)
    os.environ.pop("DKV_RSVD_MAX_RPROJ", None)
    n_moved = sum(1 for r in rows if r.get("knob_moved"))
    print("\n  pool-signature column (ADVISORY, confounded -- see pool_sig docstring):"
          " differed in %d/%d replicates." % (n_moved, len(rows)))
    print("  The knob is confirmed LIVE by colab/bench_prefill_arms_peak.py instead:")
    print("  rproj_32 is -13.4%% forward and -73 MB peak at 32k, reproducibly.")
    res = report(rows)
    res["knob_moved"] = n_moved
    print("\nJSON " + json.dumps({"summary": res, "rows": rows}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", default=os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--quant", default=None,
                    help="e.g. nf4 -- needed to fit a 7B on a 12 GB card")
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 0.9])
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return run(a.n, a.ctx, a.depths, a.seed, a.model, a.quant)


if __name__ == "__main__":
    sys.exit(main())
