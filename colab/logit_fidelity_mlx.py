#!/usr/bin/env python3
"""logit_fidelity, ported to MLX. RUN THIS ON APPLE SILICON.

WHY IT MATTERS MORE HERE THAN ON CUDA
-------------------------------------
The CUDA run answered its question but exposed a bigger one. Against a dense
control at 8k on Qwen2.5-1.5B, the CUDA DKV BASELINE sits at

    top-1 agreement 0/5    KL(dense||DKV) 10.579    dense's top-1 at rank 1255

i.e. DKV is not tracking dense at all at that operating point. That is why every
recall-based accuracy attempt on CUDA failed to discriminate anything: the arms
were being compared on top of a baseline already far off the map, and no
instrument can resolve a small change there.

MLX is the reference implementation and is expected to sit much closer to dense.
If it does, this harness has real resolving power there and can answer the
question CUDA could not: does a compression change actually move the model's
beliefs? Run the `baseline` arm first -- if MLX's baseline KL is small and its
top-1 agreement high, every subsequent comparison here is meaningful.

The CUDA baseline number is also worth confirming or refuting as a CUDA-specific
defect. If MLX's baseline is close to dense on the same prompt and model, the
CUDA gap is a CUDA bug, not the price of compression.

METHOD (identical to the CUDA version, deliberately)
----------------------------------------------------
FIRST DECODE STEP ONLY. Greedy decode diverges: once two arms pick different
tokens they condition on different prefixes, and every later comparison measures
that divergence rather than the compression. The first step is the only one
where the prefix is guaranteed identical. n comes from prompts, not steps.

Logits are captured by wrapping the module-level `_sanitize_logits`, which
`generate`'s local `sample_logits` closure calls on every step. The closure
itself is not reachable from outside, and this hook needs no change to
mlx_dkv_wrapper.py -- which is the reference implementation and is not edited.

USAGE
    python colab/logit_fidelity_mlx.py --arms dense baseline
    python colab/logit_fidelity_mlx.py --arms dense baseline basis0.50   # if
        # shared bases have been ported to MLX (see MLX_PORT_FROM_CUDA.md §1)
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")

# `basis*` arms only do something once shared bases exist on MLX. Until then
# they are inert and will silently reproduce `baseline` -- the harness says so
# rather than letting that read as "shared bases change nothing".
ARMS = {
    "dense":      {"DKV_COMPRESSED_DECODE": "0"},
    "baseline":   {},
    "basis0.50":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.50"},
    "basis0.25":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25"},
}
_ARM_KEYS = ("DKV_SHARED_BASIS", "DKV_SHARED_BASIS_FRAC", "DKV_COMPRESSED_DECODE")

QUESTION = "What is the secret passcode? Repeat it exactly."


def run_one_arm(args):
    for k in _ARM_KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[args.arm])
    # Pin the seed-sensitive draw: MLX runs the same randomised SVD as CUDA and
    # has the same +-15-point noise floor at a fixed config.
    os.environ.setdefault("DKV_SVD_SEED", "1234")
    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")

    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)

    import numpy as np
    import serving.mlx_dkv_wrapper as W
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
    from niah_recall import build_prompt

    # Wrap the module-level sanitiser that generate()'s local sample_logits
    # closure calls. Captures every step; we keep the FIRST of each generate.
    grabbed = {}
    _orig = W._sanitize_logits

    def _spy(logits, warn_owner=None):
        out = _orig(logits, warn_owner)
        if "logits" not in grabbed:
            arr = np.array(out, copy=True).reshape(-1)
            grabbed["logits"] = arr
        return out

    W._sanitize_logits = _spy

    rows = []
    try:
        w = MLXDKVWrapper(model_id=args.model)
        w.ensure_loaded()
        tok = w.tokenizer
        for depth in args.depths:
            prompt = build_prompt(tok, args.ctx, depth)
            sid = f"lf-{args.arm}-{depth}"
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
            grabbed.clear()
            w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0, query_text=QUESTION,
                       session_id=sid)
            lg = grabbed.get("logits")
            if lg is None:
                print(f"{args.arm} d={depth}: no logits captured -- did "
                      f"_sanitize_logits move?", flush=True)
                continue
            rows.append({"arm": args.arm, "depth": depth,
                         "logits": [float(x) for x in lg]})
            print(f"{args.arm} d={depth}: captured {len(lg)} logits, "
                  f"top1={int(np.argmax(lg))}", flush=True)
    finally:
        W._sanitize_logits = _orig

    with open(args.json, "w") as f:
        json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--arms", nargs="+", default=["dense", "baseline"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.arm:
        return run_one_arm(args)

    import math
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="dkv-logit-mlx-")
    per_arm = {}
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; have {list(ARMS)}")
        out = os.path.join(tmp, f"{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--model", args.model, "--ctx", str(args.ctx), "--json", out,
               "--depths"] + [str(d) for d in args.depths]
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(out):
            print(f"{arm}: FAILED (exit {p.returncode}) -- no data", flush=True)
            continue
        with open(out) as f:
            per_arm[arm] = {r["depth"]: r["logits"] for r in json.load(f)}

    if "dense" not in per_arm:
        raise SystemExit("no dense control -- nothing to compare against")

    def _softmax(v):
        m = max(v)
        e = [math.exp(x - m) for x in v]
        s = sum(e)
        return [x / s for x in e]

    print(f"\n== MLX first-step logit fidelity vs DENSE "
          f"(n={len(args.depths)} prompts, ctx={args.ctx}) ==")
    print(f"{'arm':>12} {'top1 agree':>11} {'KL(dense||arm)':>15} "
          f"{'dense-top1 rank':>16} {'top5 overlap':>13}")
    for arm in args.arms:
        if arm not in per_arm:
            continue
        agree, kls, ranks, ov, n = 0, [], [], [], 0
        for d, dl in per_arm["dense"].items():
            al = per_arm[arm].get(d)
            if al is None:
                continue
            n += 1
            pd, pa = _softmax(dl), _softmax(al)
            kls.append(sum(p * (math.log(p + 1e-12) - math.log(q + 1e-12))
                           for p, q in zip(pd, pa)))
            dt = max(range(len(dl)), key=lambda i: dl[i])
            at = max(range(len(al)), key=lambda i: al[i])
            agree += int(dt == at)
            ranks.append(sum(1 for x in al if x > al[dt]))
            top5d = sorted(range(len(dl)), key=lambda i: -dl[i])[:5]
            top5a = sorted(range(len(al)), key=lambda i: -al[i])[:5]
            ov.append(len(set(top5d) & set(top5a)))
        if not n:
            continue
        print(f"{arm:>12} {agree:>7}/{n:<3} {sum(kls)/n:>15.5f} "
              f"{sum(ranks)/n:>16.2f} {sum(ov)/n:>12.1f}/5")

    print("\nRead the BASELINE row first -- it is the whole point of running")
    print("this on MLX. CUDA's baseline is KL 10.579 with dense's top-1 at rank")
    print("1255, which leaves no resolving power for anything measured on top.")
    print("A small MLX baseline KL means (a) this instrument works here, and")
    print("(b) CUDA's gap is a CUDA defect rather than the price of compression.")
    if any(a.startswith("basis") for a in per_arm):
        print("\nNOTE: basis* arms are inert until shared bases are ported to")
        print("MLX (MLX_PORT_FROM_CUDA.md section 1). Until then they reproduce")
        print("`baseline` exactly -- which must NOT be read as 'no effect'.")


if __name__ == "__main__":
    main()
