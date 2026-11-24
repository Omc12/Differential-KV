#!/usr/bin/env python3
"""Paired difference between two RULER result stores, item by item.

run_ruler_cuda.py --compare keys arms by (model, arm/preset), so two runs of
the same arm under different switches collide there. This pairs any two stores
on their item keys and reports A - B with a paired bootstrap 95% CI, overall
and per task family.

    python benchmarks/ruler_pair.py A.jsonl B.jsonl
"""
import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL  # noqa: E402
from run_longbench_cuda import _paired_bootstrap  # noqa: E402
from run_ruler_cuda import TASK_FAMILY, official_metric_fns  # noqa: E402


def load(path, metrics):
    store = ResumableJSONL(path, config=None, strict_config=False, read_only=True)
    latest = store.load_latest()
    store.close()
    out = {}
    for r in latest.values():
        if r.get("error"):
            continue
        fam = TASK_FAMILY.get(r.get("task"))
        if fam is None:
            continue
        key = r.get("key") or (str(r["length"]) + "/" + r["task"] + "#" + str(r["idx"]))
        out[key] = (r["task"], metrics[fam]([r.get("text", "")], [r.get("outputs", [])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    metrics = official_metric_fns()
    A, B = load(args.a, metrics), load(args.b, metrics)
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit("no shared items")
    d = [A[k][1] - B[k][1] for k in keys]
    lo, hi = _paired_bootstrap(d)
    ma = sum(A[k][1] for k in keys) / len(keys)
    mb = sum(B[k][1] for k in keys) / len(keys)
    print(f"A={os.path.basename(args.a)}\nB={os.path.basename(args.b)}")
    print(f"n={len(keys)}  A {ma:.1f}  B {mb:.1f}  A-B {sum(d) / len(d):+.2f} "
          f"[{lo:+.2f}, {hi:+.2f}]  (A only {len(set(A) - set(B))}, B only {len(set(B) - set(A))})")
    by = defaultdict(list)
    for k in keys:
        by[A[k][0]].append(A[k][1] - B[k][1])
    for t in sorted(by):
        v = by[t]
        print(f"  {t:16s} n={len(v):3d}  {sum(v) / len(v):+7.2f}")


if __name__ == "__main__":
    main()
