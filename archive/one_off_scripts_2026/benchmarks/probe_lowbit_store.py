#!/usr/bin/env python3
"""
probe_lowbit_store.py -- would storing the REST of the compressed store at 4 bits
cost accuracy?

Residuals are already int4. What is not, per the allocation shapes in
_create_empty_session (block 1024, rank 32, R=128, D=128, kv_heads=4,
shared-basis off = default):

    comp_VK+VV     fp16    65,536 B   36.9%      <- basis
    comp_res_k+v_q int4    65,536 B   36.9%      <- already 4-bit
    comp_U         int8    32,736 B   18.4%      <- low-rank coefficients
    res scales/biases       8,192 B    4.6%
    anchors + min/max       4,096 B    2.4%
    mask + pos              1,535 B    0.9%
    TOTAL                 177,641 B             (173.5 KB/block/layer)

Recoding everything else at 4.5 bits/value would cut the store ~36%. But the
store was measured at 48.2 MB against a 251.9 MB fp32 decode cache at K=8, so
that is ~6% of resident bytes -- bought by putting the basis and its
coefficients, the two arrays every reconstructed row passes through, at 4 bits.

FAKE QUANT, NOT A STORAGE PATH. This round-trips the arrays through
quantize->dequantize in place, so the NUMERICS are exactly what a real 4-bit
path would produce while the bytes are unchanged. That is deliberate: measure
whether the accuracy is affordable BEFORE building the packing. Memory is
already known from the table above; it is not what this probe reports.

The failure mode this is designed to avoid is documented at
basis_group_mlx.py:342 -- on a q4_0 preset, quantisation noise takes voluntary
basis joins to zero and retained energy to 0.685 at IDENTICAL pool MB, so "a run
can look like a clean win while fidelity has collapsed". Hence: every arm
reports the perturbation it actually applied, and an arm that perturbed nothing
is called out rather than scored.

Arms (all paired -- same prompts, one process, one model load):
    none        baseline
    aa          second baseline run, the A/A noise floor
    anchors4    anchors at 4 bits (2.4% of store; sensitivity control)
    basis8      VK/VV at 8 bits
    basis4      VK/VV at 4 bits
    u8          IDENTITY control -- re-quantising int8 codes to 8 bits is exactly
                a no-op (verified: 0.000 code error, 255 levels kept), so this arm
                exercises the whole hook path and must tie `none` exactly
    u4          U at 4 bits
    both4       VK/VV and U at 4 bits  (the actual proposal)

Read `aa` and `u8` FIRST. `aa` gives the noise floor; `u8` proves the hook fires
without changing any number. If `aa` does not tie `none`, the harness has no
resolution at this sample count. If `u8` does not tie `none`, decode is not
deterministic. Either way no other arm means anything until that is fixed.

Run:
    python3 benchmarks/probe_lowbit_store.py
    python3 benchmarks/probe_lowbit_store.py --ctx 16384 --samples 8
    python3 benchmarks/probe_lowbit_store.py --arms none,aa,basis4,u4,both4
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)
sys.path.insert(0, HERE)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DKV_COMPRESSED_DECODE"] = "1"

import numpy as np
import mlx.core as mx
from serving.mlx_dkv_wrapper import MLXDKVWrapper, MLXKVBlockManager
from run_int4_residual_matrix import TASK_BUILDERS

# ── fake quant ────────────────────────────────────────────────────────────────


def fq_group_asym(x, bits, group=64):
    """Asymmetric group min/max fake-quant on the last axis.

    Same scheme as residual_quant.quantize_residuals_group_asymmetric (per-group
    min/max, scale clamped at 1e-7), so basis/U results are directly comparable
    to the residual path that already ships.
    """
    D = x.shape[-1]
    g = group if (group <= D and D % group == 0) else D
    lead = int(x.size // D)
    xg = x.reshape(lead, D // g, g).astype(np.float32)
    mn = xg.min(-1, keepdims=True)
    mxv = xg.max(-1, keepdims=True)
    qmax = (1 << bits) - 1
    scale = np.clip((mxv - mn) / qmax, 1e-7, None)
    q = np.clip(np.round((xg - mn) / scale), 0, qmax)
    return (q * scale + mn).reshape(x.shape)


def fq_int8_codes(u, bits):
    """Re-quantise int8 symmetric codes to 2^bits levels.

    comp_U already stores symmetric int8 codes with a per-block fp16 scale
    (dequantised as codes*scale/127 at mlx_dkv_wrapper.py:1674), so shrinking the
    code alphabet isolates BIT WIDTH without also changing the scheme. A real
    int4-U path would likely use finer-grained scales and do better, which makes
    this arm a LOWER BOUND on 4-bit U quality, not an estimate of it.
    """
    levels = (1 << (bits - 1)) - 1          # bits=4 -> 7  (15 symmetric levels)
    step = 127.0 / levels
    return np.clip(np.round(np.round(u.astype(np.float32) / step) * step), -127, 127)


ARM_SPEC = {
    #          basis bits, U bits, anchor bits
    "none":     (None, None, None),
    "aa":       (None, None, None),
    "anchors4": (None, None, 4),
    "basis8":   (8,    None, None),
    "basis4":   (4,    None, None),
    "u8":       (None, 8,    None),
    "u4":       (None, 4,    None),
    "both4":    (4,    4,    None),
}

_STATE = {"arm": "none", "fires": 0, "delta": []}


def _apply(session, num_layers):
    """Round-trip the store's non-residual arrays at the current arm's bit width."""
    b_bits, u_bits, a_bits = ARM_SPEC[_STATE["arm"]]
    if b_bits is None and u_bits is None and a_bits is None:
        return
    jobs = []
    if b_bits:
        jobs += [("comp_VK", b_bits, fq_group_asym), ("comp_VV", b_bits, fq_group_asym)]
    if a_bits:
        jobs += [("comp_anc_k", a_bits, fq_group_asym), ("comp_anc_v", a_bits, fq_group_asym)]
    if u_bits:
        jobs += [("comp_U", u_bits, fq_int8_codes)]

    for key, bits, fn in jobs:
        arrs = session.get(key)
        if arrs is None:
            continue
        for l in range(num_layers):
            a = arrs[l] if l < len(arrs) else None
            if a is None or a.size == 0:
                continue
            dt = a.dtype
            old = np.array(a)
            new = fn(old, bits)
            d = float(np.mean(np.abs(new - old.astype(np.float32))))
            if d > 0:
                _STATE["delta"].append(d)
            arrs[l][:] = mx.array(new).astype(dt)
    _STATE["fires"] += 1


def install_hook():
    """Fire right after prefill compression writes comp_VK / comp_VV / comp_U."""
    for name in ("compress_deferred_prefill_blocks",
                 "compress_deferred_prefill_blocks_for_layer"):
        orig = getattr(MLXKVBlockManager, name, None)
        if orig is None:
            continue

        def make(orig_fn):
            def wrapped(self, session_id, *a, **kw):
                out = orig_fn(self, session_id, *a, **kw)
                sess = self.sessions.get(session_id)
                if sess is not None:
                    _apply(sess, self.num_layers)
                return out
            return wrapped

        setattr(MLXKVBlockManager, name, make(orig))


# ── run ───────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--arms", default=",".join(ARM_SPEC))
    ap.add_argument("--tasks", default="exact_numeric,multi_key,multi_value,variable_track")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "lowbit_store_probe.json"))
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a in ARM_SPEC]
    tasks = args.tasks.split(",")

    install_hook()
    print("=" * 78)
    print("low-bit store probe (fake quant -- accuracy only, bytes unchanged)")
    print("model=%s  ctx=%d  block=%d  rank=%d  n=%d/task"
          % (args.model, args.ctx, args.block, args.rank, args.samples))
    print("arms: %s" % ", ".join(arms))
    print("=" * 78, flush=True)

    wrapper = MLXDKVWrapper(model_id=args.model,
                            config={"rank": args.rank, "block_size": args.block})

    # Build every prompt ONCE so all arms see identical inputs (paired).
    rng = random.Random(args.seed)
    cases = []
    for task in tasks:
        for i in range(args.samples):
            cases.append(TASK_BUILDERS[task](wrapper.tokenizer, args.ctx, rng))

    results = defaultdict(dict)
    for arm in arms:
        _STATE["arm"] = arm
        _STATE["fires"] = 0
        _STATE["delta"] = []
        hits = defaultdict(lambda: [0, 0])
        for n, case in enumerate(cases):
            sid = "lowbit_%s_%d" % (arm, n)
            wrapper.manager.clear_session(sid)
            if hasattr(wrapper, "_session_token_ids"):
                wrapper._session_token_ids[sid] = []
            wrapper.active_session = sid
            text = wrapper.generate(
                prompt=case["prompt"], max_new_tokens=args.max_new,
                temperature=0.0, top_p=1.0, repetition_penalty=1.0, session_id=sid,
            )
            ok = case["answer"] in text
            hits[case["task"]][0] += int(ok)
            hits[case["task"]][1] += 1
            wrapper.manager.clear_session(sid)

        tot_ok = sum(v[0] for v in hits.values())
        tot_n = sum(v[1] for v in hits.values())
        mean_delta = float(np.mean(_STATE["delta"])) if _STATE["delta"] else 0.0
        results[arm] = {
            "per_task": {k: "%d/%d" % (v[0], v[1]) for k, v in hits.items()},
            "total": "%d/%d" % (tot_ok, tot_n),
            "pct": round(100.0 * tot_ok / max(1, tot_n), 1),
            "hook_fires": _STATE["fires"],
            "mean_abs_perturbation": mean_delta,
        }

        flag = ""
        if arm == "u8":
            # Re-quantising int8 codes to 8 bits is EXACTLY identity (verified:
            # 0.000 code error, all 255 levels preserved). So this arm is a second
            # A/A control that costs nothing: it must tie `none` exactly, and the
            # hook must still fire. If it does not tie, the harness is not
            # deterministic and no other arm can be read.
            flag = "  (identity control -- must tie `none`)"
            if _STATE["fires"] == 0:
                flag = "  <-- HOOK NEVER FIRED: the hook site is wrong, fix before reading on"
        elif arm not in ("none", "aa"):
            if _STATE["fires"] == 0:
                flag = "  <-- HOOK NEVER FIRED: this arm is a copy of the baseline"
            elif mean_delta == 0.0:
                flag = "  <-- PERTURBED NOTHING: quantisation was a no-op here"
        print("  %-9s %-8s (%5.1f%%)  fires=%-4d perturb=%.5f%s"
              % (arm, results[arm]["total"], results[arm]["pct"],
                 _STATE["fires"], mean_delta, flag), flush=True)

    print("\n" + "-" * 78)
    print("per-task:")
    for arm in arms:
        print("  %-9s %s" % (arm, results[arm]["per_task"]))

    print("\n" + "-" * 78)
    print("HOW TO READ THIS")
    if "none" in results and "aa" in results:
        spread = abs(results["none"]["pct"] - results["aa"]["pct"])
        print("  A/A spread (none vs aa): %.1f pts -- that is this harness's noise" % spread)
        print("  floor at n=%d. Any arm within %.1f pts of `none` is UNRESOLVED, not"
              % (len(cases), spread))
        print("  'free'. If the spread is large, raise --samples before reading on.")
    print("  Any arm whose perturbation is 0.00000 measured nothing -- that is the")
    print("  false-win signature from basis_group_mlx.py:342, not a good result.")
    print("  A real 4-bit basis/U path would buy ~36%% of the compressed store, which")
    print("  is ~6%% of resident bytes. Weigh whatever accuracy you see against that,")
    print("  not against the 3.56x the residual packing gets.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": dict(results)}, f, indent=2)
    print("\nsaved: %s" % args.out)


if __name__ == "__main__":
    main()
