#!/usr/bin/env python3
"""Several needle cases in ONE process, asserting on an EXACT STRING.

WHY THIS EXISTS
---------------
The MLX port found four defects immediately with an instrument of this shape,
while the logit harness -- which bottoms out at one fp16 ULP -- called every arm
identical, correctly and uselessly. CUDA had no equivalent: validate_cuda_dkv.py
runs each case in its own process AND applies BEST_DECODE_DEFAULTS, so it could
not see either (a) state that leaks between requests, or (b) any defect that only
appears when a caller does NOT apply the serving defaults.

(b) was not hypothetical. With DKV_SPARSE_BIAS at its library default the decode
forward took the COMBINED kernel branch, which hands _remat_attend a dense window
rotated into a different frame from the compressed half it materialises. Measured
at 8k on Qwen2.5-1.5B, first decode step against a plain transformers control:
KL 11.76 with top-1 agreement 0/5, and the needle lost outright. With
DKV_SPARSE_BIAS=auto -- which BEST_DECODE_DEFAULTS sets, and which is therefore
all validate_cuda_dkv.py ever exercised -- the same build reads KL 0.00125 and
5/5. Nothing raised, no shape changed, and the pool reported the same block count
either way.

SO THIS DELIBERATELY DOES NOT APPLY BEST_DECODE_DEFAULTS. Running the library as
a caller gets it out of the box is the whole point; pass --serving-defaults to
measure the other configuration.

USAGE
    python colab/needle_suite_cuda.py
    python colab/needle_suite_cuda.py --arms default noremat sparsebias
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")

ARMS = {
    "default":    {},                                # the library as shipped
    "noremat":    {"DKV_REMAT_CACHE": "0"},
    "sparsebias": {"DKV_SPARSE_BIAS": "auto"},       # what the serving defaults set
}
_KEYS = ("DKV_REMAT_CACHE", "DKV_SPARSE_BIAS")

# (ctx, depth). Several depths at one context and one longer context: the
# combined-branch defect was depth-INDEPENDENT, which is what said it was
# reconstruction rather than routing.
CASES = [(8192, 0.1), (8192, 0.5), (8192, 0.9), (16384, 0.5)]


def run_arm(args):
    for k in _KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[args.arm])
    os.environ.setdefault("DKV_RSVD_SEED", "1234")
    os.environ.setdefault("DKV_SVD_SEED", "1234")
    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    if args.serving_defaults:
        from serving.decode_config import BEST_DECODE_DEFAULTS
        for k, v in BEST_DECODE_DEFAULTS.items():
            os.environ.setdefault(k, v)

    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import NEEDLE, QUESTION, build_prompt

    w = DKVHFWrapper(model_id=args.model,
                     config={"quantization": None, "rank": 32, "block_size": 256,
                             "micro_block_size": 256, "preset": "mid"})
    w.ensure_loaded()
    tok = w.tokenizer
    rows = []
    # ONE wrapper, every case, IN ORDER -- so anything that leaks between
    # requests (pool slots, basis groups, routing state) is reachable here and
    # is not reachable from a per-case process.
    for ctx, depth in CASES:
        prompt = build_prompt(tok, ctx, depth)
        sid = f"needle-{args.arm}-{ctx}-{depth}"
        w.active_session = sid
        out = w.generate(prompt, max_new_tokens=24, temperature=0.0, top_p=1.0,
                         repetition_penalty=1.0, query_text=QUESTION)
        # ISOLATE THE COMPLETION EXACTLY. generate() returns prompt+completion,
        # and the prompt contains the needle, so the whole string cannot be
        # searched. Character slicing when the prompt is a literal prefix is
        # exact; re-tokenising is NOT -- decode(encode(x)) != x here, and the
        # round trip clipped real answers to '-DELTA' and scored them FAIL.
        if out.startswith(prompt):
            comp = out[len(prompt):]
        else:
            pn = len(tok(prompt).input_ids)
            oid = tok(out).input_ids
            comp = tok.decode(oid[max(0, pn - 4):]) if len(oid) > pn else ""
        ok = NEEDLE in comp
        nb = 0
        try:
            sess = w.manager.sessions.get(sid) or {}
            nb = int(sum(sess.get("num_blocks") or []))
        except Exception:                                        # noqa: BLE001
            nb = -1
        rows.append({"ctx": ctx, "depth": depth, "ok": bool(ok),
                     "blocks": nb, "text": comp[:80]})
        print(f"  [{'PASS' if ok else 'FAIL'}] {ctx//1024}k@{depth} "
              f"blocks={nb} -> {comp[:60]!r}", flush=True)
        try:
            w.clear_session(sid)
        except Exception:                                        # noqa: BLE001
            pass
    with open(args.json, "w") as f:
        json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--arm", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--serving-defaults", action="store_true",
                    help="apply BEST_DECODE_DEFAULTS; OFF by default, because "
                         "running without them is what this suite exists to cover")
    args = ap.parse_args()
    if args.arm:
        return run_arm(args)

    import tempfile
    tmp = tempfile.mkdtemp(prefix="dkv-needle-")
    res = {}
    for arm in args.arms:
        out = os.path.join(tmp, f"{arm}.json")
        print(f"\n== arm {arm} "
              f"({'with' if args.serving_defaults else 'WITHOUT'} serving defaults) ==",
              flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--model", args.model, "--json", out]
        if args.serving_defaults:
            cmd.append("--serving-defaults")
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(out):
            print(f"{arm}: FAILED (exit {p.returncode})", flush=True)
            continue
        with open(out) as f:
            res[arm] = json.load(f)

    print(f"\n{'arm':>12} " + " ".join(f"{str(c//1024)+'k@'+str(d):>9}"
                                       for c, d in CASES) + "   total")
    for arm, rows in res.items():
        marks = " ".join(f"{('PASS' if r['ok'] else 'FAIL'):>9}" for r in rows)
        n = sum(1 for r in rows if r["ok"])
        print(f"{arm:>12} {marks}   {n}/{len(rows)}")


if __name__ == "__main__":
    main()
