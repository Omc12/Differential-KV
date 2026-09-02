#!/usr/bin/env python3
"""Audit every result file before it is allowed into a table.

WHY THIS EXISTS
---------------
This campaign has produced six separate contaminations, every one silent and
every one caught only after it had already burned GPU time:

  1. reasoning models scored on their <think> scratchpad (LongBench)
  2. the same, again, on RULER -- the fix did not reach it, because RULER
     prompts carry no chat template
  3. DKV run on a decode branch production never executes (library defaults)
  4. SnapKV/H2O handicapped ~5x by a whole-model eager load
  5. a checkpoint resume that merged pre- and post-attention-scale-fix rows
  6. two processes writing one file while sharing one GPU

Each was found by looking. That is not a process. This checks every file
mechanically, so a contaminated arm is caught before it reaches a table rather
than after it reaches a draft.

Exit code is non-zero if anything FAILs, so it can gate a report.

USAGE
    python benchmarks/validate_results.py
    python benchmarks/validate_results.py --dir paper/results/longbench
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from code_fingerprint import decode_fingerprint                  # noqa: E402

FAIL, WARN, OK = "FAIL", "warn", "ok"


def _load(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue                 # torn final line: a power cut
            raise
    return recs


def audit(path):
    """Return (status, [messages]) for one result file."""
    msgs = []
    status = OK

    def bad(m):
        nonlocal status
        status = FAIL
        msgs.append("FAIL " + m)

    def warn(m):
        nonlocal status
        if status != FAIL:
            status = WARN
        msgs.append("warn " + m)

    meta_p = path + ".meta.json"
    meta = {}
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        bad("no .meta.json — the run's configuration is unrecorded")

    recs = _load(path)
    if not recs:
        bad("no records")
        return status, msgs, meta, recs

    arm = meta.get("arm", "?")

    # ── 1. duplicate keys: the signature of a merged/resumed-across-change run
    keys = [r.get("key") for r in recs if r.get("key")]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    if dupes:
        warn(f"{len(dupes)} duplicate key(s) — later rows win, but a resume "
             f"across a code change looks exactly like this "
             f"(e.g. {dupes[:2]})")

    # ── 2. errors
    errs = [r for r in recs if r.get("error")]
    if errs:
        kinds = Counter(str(r["error"]).split(":")[0] for r in errs)
        warn(f"{len(errs)} error row(s), excluded from scoring: {dict(kinds)}")

    live = [r for r in recs if not r.get("error")]

    # Files that record no text at all (the ladder measures memory, not
    # answers) must skip every text-based check below, or they report 100%
    # empty outputs and nothing useful.
    has_text = any("text" in r for r in live)

    # ── 3. reasoning-mode contamination — the one that has recurred
    thinking_cfg = meta.get("thinking")
    # ONLY on rows whose prompt echo was cleanly stripped. On a row where it was
    # not, "<think>" is a chat-template marker inside the ECHOED PROMPT, not the
    # model reasoning -- verified on granite dkv_high/dkv_mid, where the two
    # flagged rows were exactly the two echo-unclean ones and their text was
    # Project Gutenberg source, not a scratchpad. Counting those would send a
    # clean arm back for a two-hour re-run it does not need.
    n_think = sum(1 for r in live
                  if r.get("prompt_echo_clean") is not False
                  and "<think" in (r.get("text") or "").lower())
    if has_text and n_think:
        frac = n_think / max(1, len(live))
        bad(f"{n_think}/{len(live)} ({frac:.0%}) outputs contain '<think>' — "
            f"the generation budget is being spent on reasoning, not answers")
    if thinking_cfg is None and has_text:
        warn("config does not record `thinking`; cannot prove the reasoning "
             "block was disabled")
    elif thinking_cfg:
        warn("`thinking` is TRUE for this run")

    # ── 4. prompt echo (DKV's generate returns prompt+completion)
    echoed = [r for r in live if r.get("prompt_echo_clean") is False]
    if echoed:
        warn(f"{len(echoed)} row(s) where the prompt echo could not be cleanly "
             f"stripped; their text was truncated to the generation budget")

    # ── 5. timing comparability
    fused = {r.get("inductor_fused") for r in live if "inductor_fused" in r}
    if len(fused) > 1:
        bad(f"mixed inductor_fused within one file {fused} — the latency rows "
            f"were measured on two different code paths")
    elif fused == {False}:
        warn("inductor_fused is False: Inductor never compiled, so latency "
             "here UNDERSTATES DKV. Quality is unaffected.")

    attn = {r.get("attn_eager") for r in live if "attn_eager" in r}
    if True in attn:
        warn("attn_eager is True: prefill ran on eager attention, so TTFT is "
             "not comparable to the SDPA arms")

    # ── 6. DKV arms must match the current decode arithmetic
    if arm == "dkv":
        rev, cur = meta.get("dkv_decode_rev"), decode_fingerprint()
        if rev is None:
            bad("DKV arm with no dkv_decode_rev — predates the fingerprint "
                "guard, so it may have been measured on older decode code")
        elif rev != cur:
            bad(f"dkv_decode_rev {rev} != current {cur} — measured on "
                f"different decode arithmetic")
        if meta.get("decode_defaults") != "serving":
            bad("DKV arm not run under the shipped serving decode defaults")

    if meta.get("prefill_attn") is None and has_text:
        warn("config does not record `prefill_attn`")

    # ── 7. empty outputs
    empty = sum(1 for r in live if not (r.get("text") or "").strip())
    if has_text and empty > max(2, 0.05 * len(live)):
        warn(f"{empty}/{len(live)} empty outputs")

    return status, msgs, meta, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="*",
                    default=["paper/results/longbench", "paper/results/ruler",
                             "paper/results/ladder"])
    args = ap.parse_args()

    paths = []
    for d in args.dir:
        paths += sorted(glob.glob(os.path.join(REPO, d, "*.jsonl")))
    paths = [p for p in paths if "_discarded" not in p and not p.endswith(".bak")]
    if not paths:
        raise SystemExit("no result files found")

    print(f"decode fingerprint (current): {decode_fingerprint()}\n")
    tally = Counter()
    for p in paths:
        status, msgs, meta, live = audit(p)
        tally[status] += 1
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {os.path.relpath(p, REPO)}")
        print(f"          arm={meta.get('arm','?'):<14} rows={len(live):<5} "
              f"thinking={meta.get('thinking')} "
              f"decode_defaults={meta.get('decode_defaults')}")
        for m in msgs:
            print(f"          {m}")
    print(f"\n{tally[OK]} ok, {tally[WARN]} warn, {tally[FAIL]} FAIL")
    if tally[FAIL]:
        print("\nA FAIL means that file must not appear in a results table "
              "until it is re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
