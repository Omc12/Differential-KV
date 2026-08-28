#!/usr/bin/env python3
"""Measure what the SRL/storage branch costs and buys, on a real model.

Each arm runs the SAME prompt through the wrapper's own generate() and reports
pool VRAM, needle recall, whether the generated text is byte-identical to the
baseline arm, mean residual fill, and the shared-basis sharing factor and
retained energy.

WHY IT IS BUILT THIS WAY
------------------------
This repo has had to retract several unmeasured or under-powered claims (see
ACTIVE_RUNTIME/docs/cuda_port_record.md, and the seed-noise warning carried
forward into ACTIVE_RUNTIME/docs/mlx_work_record.md: the randomised SVD's seed alone moves
synthesis 30 points at a fixed config). So:

  * The seed-sensitive quantity (DKV_RSVD_SEED) is PINNED across arms, so a
    difference cannot be the projection draw.
  * The pool budget is PINNED, so block sizing (and therefore routing) cannot
    depend on what the previous arm left allocated -- the isolation fix the
    test suite's conftest documents.

ONE ARM PER PROCESS, and that is a real limitation worth stating. The pool
reads DKV_SHARED_BASIS at construction, so an arm cannot be applied to a pool
that already exists; and loading a second model into one process aborts the
interpreter here (a native crash inside the allocator, not a Python error).
So arms are compared ACROSS processes, which is weaker than the paired,
interleaved form `bench_decode_paired.py` uses:

  * VRAM is unaffected -- it is a deterministic function of the config, and a
    fresh process is if anything cleaner (no allocator carryover).
  * TIMINGS are NOT paired and carry full between-process variance. `gen s` is
    whole-call wall time (prefill AND decode together) and is indicative only.
    There is deliberately no tok/s column -- see the note in run_arm. Use
    bench_decode_paired.py to establish a timing difference.
  * Recall is checked in the GENERATED tokens only, never the prompt-inclusive
    response, which would false-positive on the planted needle.
  * VRAM is reported from the POOL, not from process totals -- the pool is the
    only line KV compression can move, and measuring against the process is the
    denominator trap item 5c warns about.
  * n is printed with every number. A single greedy run is ONE sample; it is
    reported as one, not as a measurement.

USAGE
    python colab/srl_tradeoffs.py --model Qwen/Qwen2.5-1.5B-Instruct --ctx 8192
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")

NEEDLE = "OMEGA-7741-DELTA"
QUESTION = "What is the secret passcode? Repeat it exactly."


def _pool_mb(mgr):
    p = getattr(mgr, "native_pool", None)
    if p is None:
        return 0.0
    try:
        return float(p._pool_mb())
    except Exception:                                            # noqa: BLE001
        return 0.0


def _basis_stats(mgr):
    p = getattr(mgr, "native_pool", None)
    if p is None or not hasattr(p, "basis_stats"):
        return {"enabled": False}
    try:
        return p.basis_stats()
    except Exception:                                            # noqa: BLE001
        return {"enabled": False}


def _residual_fill(mgr):
    """Mean residual slots actually filled per written block.

    The anchor-delta budget's DIRECT effect: it can only ever raise a block's
    budget, so if this does not move, the change did nothing and any downstream
    difference is noise.
    """
    p = getattr(mgr, "native_pool", None)
    if p is None or getattr(p, "residual_K_positions", None) is None:
        return 0.0
    import torch
    with torch.no_grad():
        live = (p.seq_lens[:p.current_blocks] > 0)
        if int(live.sum()) == 0:
            return 0.0
        filled = (p.residual_K_positions[:p.current_blocks] >= 0).sum(dim=1).float()
        return float(filled[live].mean())


def run_arm(wrapper, tok, mgr, model, name, ctx, depth, gen, warmup=1):
    """One prompt through the WRAPPER'S OWN generate().

    Driving `model(...)` by hand -- chunked prefill with explicit position ids,
    then argmax decode -- looks like what benchmarks/niah_recall.py does, but
    that harness targets MLX. On the HF/CUDA path it produces word salad from
    BOTH the DKV and the dense arm, because the KV cache the wrapper owns is
    never threaded through (the same class of bug as commit a22eddc5, "the
    batch engine owns a KV cache -- word salad becomes real text").

    A harness whose dense control is also garbage is measuring nothing, and it
    fails silently: the arms still differ from each other in tidy, plausible
    ways. Hence the dense arm, and hence going through generate().
    """
    import torch
    sys.path.insert(0, BENCH)
    from niah_recall import build_prompt

    prompt = build_prompt(tok, ctx, depth)
    sid = f"arm-{name}-{depth}"
    try:
        wrapper.clear_session(sid)
    except Exception:                                            # noqa: BLE001
        pass
    wrapper.active_session = sid

    def _gen(n):
        try:
            wrapper.clear_session(sid)
        except Exception:                                        # noqa: BLE001
            pass
        wrapper.active_session = sid
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        o = wrapper.generate(prompt, max_new_tokens=n, temperature=0.0,
                             top_p=1.0, repetition_penalty=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        el = time.perf_counter() - t
        s = o if isinstance(o, str) else (
            o.get("text", "") if isinstance(o, dict) else str(o))
        return s, el

    # NO DECODE-TPS COLUMN. generate() returns text only, and differencing a
    # 1-token run against an n-token run does not isolate decode here because
    # generation STOPS EARLY at EOS -- both runs end up the same length, the
    # difference collapses to noise, and the quotient explodes (a first attempt
    # printed 7.98e12 tok/s). Wall time for the whole call is reported instead,
    # which is honest about conflating prefill and decode.
    #
    # For a real timing comparison use colab/bench_decode_paired.py: it
    # interleaves arms in one process and reports a paired statistic with a
    # confidence interval, which is what a throughput claim needs.
    full, t_full = _gen(gen)

    # STRIP THE PROMPT. wrapper.generate() decodes `generated`, which despite
    # the name holds prompt+new tokens, so the returned string contains the
    # planted needle verbatim. Scoring recall on it makes EVERY arm pass --
    # including one that emits nothing at all. niah_recall.py's docstring warns
    # about exactly this ("checking the needle in the GENERATED tokens only ...
    # that would false-positive on the planted needle") and a first run of this
    # harness reported a tidy 3/3 for all seven arms because of it.
    #
    # The question is the last thing in the prompt, so everything after its
    # final occurrence is the completion.
    text = full.split(QUESTION)[-1] if QUESTION in full else full
    n_new = len(tok.encode(text, add_special_tokens=False))

    pool_mb = _pool_mb(mgr)
    bstats = _basis_stats(mgr)
    res_fill = _residual_fill(mgr)

    try:
        wrapper.clear_session(sid)
    except Exception:                                            # noqa: BLE001
        pass
    return {
        "arm": name, "depth": depth,
        "pool_mb": round(pool_mb, 2),
        "gen_s": round(t_full, 2),
        "n_tok": n_new,
        "recall": int(NEEDLE in text),
        "res_fill": round(res_fill, 2),
        "basis": bstats,
        "text": text,
        "sample": text[:70].replace("\n", " "),
    }


ARMS = {
    # The DENSE control. This repo's own guidance is to run one alongside every
    # time -- it is what distinguishes "DKV regressed" from "this harness is
    # measuring nothing". Without it, every DKV arm could be producing garbage
    # and the comparison between them would still look tidy.
    "dense":         {"DKV_COMPRESSED_DECODE": "0"},
    "baseline":      {},
    "basis0.50":     {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.50"},
    "basis0.25":     {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25"},
    "basis0.125":    {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.125"},
    "anchordelta":   {"DKV_RESIDUAL_ANCHOR_DELTA": "1"},
    "basis+anchor":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25",
                      "DKV_RESIDUAL_ANCHOR_DELTA": "1"},
    # DKV_TOPK_BLOCKS=0 is attend-every-block, the routing reference.
    "attendall":     {"DKV_TOPK_BLOCKS": "0"},
}
_ARM_KEYS = ("DKV_SHARED_BASIS", "DKV_SHARED_BASIS_FRAC",
             "DKV_RESIDUAL_ANCHOR_DELTA", "DKV_TOPK_BLOCKS",
             "DKV_COMPRESSED_DECODE")


def _pin_env():
    # PIN the seed-sensitive draw. Without this a difference between arms could
    # be the randomised-SVD projection rather than the feature -- the failure
    # mode MLX_PORT's header is entirely about.
    os.environ.setdefault("DKV_RSVD_SEED", "1234")
    os.environ.setdefault("DKV_SVD_SEED", "1234")
    # setdefault, so the `dense` arm's explicit "0" survives.
    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    # The pool budget must not float with free VRAM, or block sizing (and
    # therefore routing) varies between arms for reasons unrelated to the arm.
    # Same reasoning as the test suite's conftest.
    os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")


def run_one_arm(args):
    """Body of a single-arm subprocess: load, measure every depth, emit JSON."""
    # Order matters: clear every arm key, apply THIS arm, then pin the rest
    # with setdefault. Pinning first would be undone by the pop loop, leaving
    # DKV_COMPRESSED_DECODE unset -- which is not "1", it is "auto", and auto
    # runs the DENSE path below 8k. Every DKV arm would then have been
    # measuring dense and the table would have looked perfectly consistent.
    for k in _ARM_KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[args.arm])
    _pin_env()

    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper

    wrapper = DKVHFWrapper(model_id=args.model,
                           config={"quantization": None, "rank": 32,
                                   "block_size": 256, "micro_block_size": 256,
                                   "preset": "mid"})
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model

    rows = []
    for depth in args.depths:
        r = run_arm(wrapper, tok, mgr, model, args.arm, args.ctx, depth, args.gen)
        rows.append(r)
        b = r["basis"]
        share = (f" share={b.get('sharing_factor', 0):.1f}x "
                 f"kept={b.get('mean_kept', 1):.3f} forced={b.get('forced', 0)}"
                 if b.get("enabled") else "")
        print(f"{args.arm:>14} d={depth:<4} pool={r['pool_mb']:>7.1f}MB "
              f"gen={r['gen_s']:>6.2f}s ntok={r['n_tok']:>3} "
              f"recall={r['recall']} resfill={r['res_fill']:>5.1f}{share}\n"
              f"{'':>14}   -> {r['sample']!r}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--json", default="")
    ap.add_argument("--arm", default="", help="internal: run this ONE arm in-process")
    args = ap.parse_args()


    if args.arm:
        return run_one_arm(args)

    import subprocess
    import tempfile
    rows = []
    tmpdir = tempfile.mkdtemp(prefix="dkv-tradeoff-")
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; have {list(ARMS)}")
        out = os.path.join(tmpdir, f"{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__),
               "--arm", arm, "--model", args.model, "--ctx", str(args.ctx),
               "--gen", str(args.gen), "--json", out,
               "--depths"] + [str(d) for d in args.depths]
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(out):
            # Report the gap rather than dropping the arm silently: a missing
            # row in a comparison table reads as "no difference", which is the
            # opposite of "did not run".
            print(f"{arm:>14} FAILED (exit {p.returncode}) -- no data for this arm",
                  flush=True)
            continue
        with open(out) as f:
            rows.extend(json.load(f))

    print("\n== summary (n=%d prompts per arm, one process per arm) ==" % len(args.depths))
    print(f"{'arm':>14} {'pool MB':>9} {'vs base':>8} {'gen s':>8} "
          f"{'ntok':>6} {'recall':>7} {'resfill':>8} {'share':>7} {'kept':>7} {'same':>6}")
    # Both reference columns are against the DKV `baseline` arm, never against
    # whichever arm happens to be listed first -- with `dense` first, `vs base`
    # divided by a 0.0 MB pool and printed +9.1e12%.
    _ref = "baseline" if any(r["arm"] == "baseline" for r in rows) else args.arms[0]
    _ref_rows = [r for r in rows if r["arm"] == _ref]
    base = (sum(r["pool_mb"] for r in _ref_rows) / len(_ref_rows)) if _ref_rows else None
    # Byte-identity of the COMPLETION against the baseline arm. This is the
    # method MLX_PORT item 1 used to settle whether routing changes anything,
    # and it is the sharpest available signal when recall alone does not
    # separate the arms: identical text means the change is inert on this
    # prompt, different text means it is not, with no threshold to argue about.
    base_text = {r["depth"]: r["text"] for r in _ref_rows}
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        same = sum(1 for r in sub if base_text.get(r["depth"]) == r["text"])
        mb = sum(r["pool_mb"] for r in sub) / len(sub)
        pf = sum(r["gen_s"] for r in sub) / len(sub)
        tp = sum(r["n_tok"] for r in sub) / len(sub)
        rc = sum(r["recall"] for r in sub)
        rf = sum(r["res_fill"] for r in sub) / len(sub)
        bs = [r["basis"] for r in sub if r["basis"].get("enabled")]
        sh = (sum(b.get("sharing_factor", 0) for b in bs) / len(bs)) if bs else 0.0
        kp = (sum(b.get("mean_kept", 1.0) for b in bs) / len(bs)) if bs else 1.0
        if base is None:
            base = mb
        print(f"{arm:>14} {mb:>9.1f} {100*(mb-base)/max(base,1e-9):>+7.1f}% "  # noqa: E501
              f"{pf:>8.2f} {tp:>6.1f} {rc:>4}/{len(sub):<2} {rf:>8.1f} "
              f"{sh:>6.1f}x {kp:>7.3f} {same:>3}/{len(sub):<2}")
    print("\nTimings are ACROSS processes and not paired -- indicative only. "
          "VRAM is a deterministic function of the config and is exact.")
    print("`same` = generated text byte-identical to the first arm.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
