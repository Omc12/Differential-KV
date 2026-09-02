#!/usr/bin/env python3
"""At a fixed VRAM budget, what context can each method serve?

WHY THIS FRAMING
----------------
Every other table here fixes the CONTEXT and compares memory and quality. A
deployment does the opposite: the card is fixed, and the question is what you
can run on it. That is the MLSys-native reading of the same data, and it is the
one that makes the ladder result concrete -- "DKV reaches 98,304 where the rest
stop at 49,152" becomes "on a 12 GB card DKV serves 2x the context".

Derived entirely from measurements already on disk (the context ladder), so it
adds no GPU time and cannot disagree with the tables it summarises.

Two things it will NOT do:

  * extrapolate past a measured spill. If an arm's largest clean rung is
    49,152, that is what it reports -- not a fitted projection. The fit is
    shown separately and labelled, because a ceiling predicted from a slope is
    a different kind of claim from a ceiling that was run.
  * hide the architecture split. Granite and the hybrids behave differently and
    the table says so per model rather than averaging them into one number.

USAGE
    python benchmarks/iso_memory_table.py
    python benchmarks/iso_memory_table.py --budgets 6 8 10 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL                            # noqa: E402


def load_ladder(paths):
    """model -> arm -> [(ctx, peak_gb, reserved_gb, status)] for clean points."""
    out = defaultdict(lambda: defaultdict(list))
    for p in paths:
        meta = {}
        mp = p + ".meta.json"
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        model = meta.get("model") or os.path.basename(p).split("_")[0]
        st = ResumableJSONL(p, config=None, strict_config=False, read_only=True)
        rows = list(st.load_latest().values())
        st.close()
        for r in rows:
            if r.get("ctx_actual") and r.get("peak_gb"):
                out[model][r.get("arm", "?")].append(
                    (r["ctx_actual"], r["peak_gb"],
                     r.get("peak_reserved_gb") or r["peak_gb"],
                     r.get("status", "?")))
    for m in out:
        for a in out[m]:
            out[m][a].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", nargs="*",
                    default=["paper/results/ladder/*_mid_nf4.jsonl"])
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[6.0, 8.0, 10.0, 12.0])
    args = ap.parse_args()

    paths = []
    for pat in args.ladder:
        paths += sorted(glob.glob(os.path.join(REPO, pat)))
    paths = [p for p in paths if "DIAG" not in p]
    if not paths:
        raise SystemExit("no ladder files found")

    data = load_ladder(paths)
    for model in sorted(data):
        arms = data[model]
        print(f"\n=== {model} — largest context served within a VRAM budget ===")
        print("(measured rungs only; reserved memory is the binding figure "
              "because that is what the allocator takes from the card)")
        print(f"{'arm':>14} " + " ".join(f"{b:>8.0f} GB" for b in args.budgets)
              + f" {'max clean':>11}")
        print("-" * (15 + 11 * len(args.budgets) + 12))
        for arm in sorted(arms, key=lambda a: (a != "dense", a)):
            pts = [(c, pk, rv) for (c, pk, rv, st) in arms[arm] if st == "ok"]
            cells = []
            for b in args.budgets:
                fit = [c for (c, pk, rv) in pts if rv <= b]
                cells.append(f"{max(fit):>11,}" if fit else f"{'—':>11}")
            best = max((c for (c, _, _) in pts), default=0)
            print(f"{arm:>14} " + " ".join(cells) + f" {best:>11,}")

        # Ratio against dense, at the largest budget where dense fits at all.
        dense_pts = [(c, rv) for (c, pk, rv, st) in arms.get("dense", [])
                     if st == "ok"]
        if dense_pts:
            dmax = max(c for c, _ in dense_pts)
            for arm in sorted(arms):
                if arm == "dense":
                    continue
                pts = [c for (c, pk, rv, st) in arms[arm] if st == "ok"]
                if pts and dmax:
                    print(f"    {arm} / dense max context: {max(pts) / dmax:.2f}x")

    print("\nNOT extrapolated: every figure is a rung that actually ran clean. "
          "An arm's column is blank where no measured rung fits that budget, "
          "rather than being filled in from a slope.")


if __name__ == "__main__":
    main()
