#!/usr/bin/env python3
"""Measure what the SRL/storage branch costs and buys, on a real model.

Every arm runs the SAME prompt through the SAME wrapper in ONE process, so the
numbers are paired: pool VRAM, prefill/compress wall time, decode tok/s, and
needle recall, plus the shared-basis sharing factor and mean retained energy.

WHY IT IS BUILT THIS WAY
------------------------
This repo has had to retract several unmeasured or under-powered claims (see
MLX_PORT_FROM_CUDA.md, whose item 10 is a full retraction and whose header
warns that the randomised SVD's seed alone moves synthesis 30 points). So:

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
  * TIMINGS are NOT paired and carry full between-process variance. Treat the
    tps and prefill columns as indicative, not as resolved differences, and
    use bench_decode_paired.py if a timing difference needs to be established.
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
    import numpy as np
    import torch
    sys.path.insert(0, BENCH)
    from niah_recall import build_prompt

    prompt = build_prompt(tok, ctx, depth)
    ids = tok.encode(prompt)
    # The reference NIAH harness is MLX, where memory is unified and a CPU
    # tensor reaches the embedding fine. On CUDA it does not -- input_ids must
    # be on the model's device or embed_tokens raises before any DKV code runs.
    dev = next(model.parameters()).device

    def _t(x):
        return torch.tensor(x, dtype=torch.long, device=dev)

    sid = f"arm-{name}"
    mgr.clear_session(sid)
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    model._dkv_session_ids = [sid]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_pre = time.perf_counter()
    out = None
    for cs in range(0, len(ids), 512):
        ch = ids[cs:cs + 512]
        out = model(_t([ch]), _t([list(range(cs, cs + len(ch)))]))
        mgr.compress_deferred_prefill_blocks(sid)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_pre

    pool_mb = _pool_mb(mgr)
    bstats = _basis_stats(mgr)
    res_fill = _residual_fill(mgr)

    logits = out.logits[0, -1].float().cpu().numpy()
    cur = len(ids)
    generated = []
    for _ in range(warmup):
        nid = int(np.argmax(logits))
        generated.append(nid)
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        out = model(_t([[nid]]), _t([[cur]]))
        logits = out.logits[0, -1].float().cpu().numpy()
        cur += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    for _ in range(gen):
        nid = int(np.argmax(logits))
        generated.append(nid)
        steps += 1
        if NEEDLE in tok.decode(generated):
            break
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        out = model(_t([[nid]]), _t([[cur]]))
        logits = out.logits[0, -1].float().cpu().numpy()
        cur += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    text = tok.decode(generated)
    mgr.clear_session(sid)
    return {
        "arm": name, "depth": depth,
        "pool_mb": round(pool_mb, 2),
        "prefill_s": round(prefill_s, 3),
        "tps": round(steps / dt, 2) if dt > 0 else 0.0,
        "recall": int(NEEDLE in text),
        "res_fill": round(res_fill, 2),
        "basis": bstats,
        "text": text,
        "sample": text[:70].replace("\n", " "),
    }


ARMS = {
    "baseline":      {},
    "basis0.50":     {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.50"},
    "basis0.25":     {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25"},
    "basis0.125":    {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.125"},
    "anchordelta":   {"DKV_RESIDUAL_ANCHOR_DELTA": "1"},
    "basis+anchor":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25",
                      "DKV_RESIDUAL_ANCHOR_DELTA": "1"},
}
_ARM_KEYS = ("DKV_SHARED_BASIS", "DKV_SHARED_BASIS_FRAC",
             "DKV_RESIDUAL_ANCHOR_DELTA", "DKV_LEARNED_ROUTER",
             "DKV_LEARNED_ROUTER_DYNK")


def _pin_env():
    # PIN the seed-sensitive draw. Without this a difference between arms could
    # be the randomised-SVD projection rather than the feature -- the failure
    # mode MLX_PORT's header is entirely about.
    os.environ.setdefault("DKV_RSVD_SEED", "1234")
    os.environ.setdefault("DKV_SVD_SEED", "1234")
    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    # The pool budget must not float with free VRAM, or block sizing (and
    # therefore routing) varies between arms for reasons unrelated to the arm.
    # Same reasoning as the test suite's conftest.
    os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")


def run_one_arm(args):
    """Body of a single-arm subprocess: load, measure every depth, emit JSON."""
    _pin_env()
    for k in _ARM_KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[args.arm])

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
              f"prefill={r['prefill_s']:>6.2f}s tps={r['tps']:>6.1f} "
              f"recall={r['recall']} resfill={r['res_fill']:>5.1f}{share}",
              flush=True)
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
    print(f"{'arm':>14} {'pool MB':>9} {'vs base':>8} {'prefill s':>10} "
          f"{'tps':>7} {'recall':>7} {'resfill':>8} {'share':>7} {'kept':>7} {'same':>6}")
    base = None
    # Byte-identity of the GENERATED text against the baseline arm. This is the
    # method MLX_PORT item 1 used to settle whether routing changes anything,
    # and it is the sharpest available signal when recall alone does not
    # separate the arms: identical text means the change is inert on this
    # prompt, different text means it is not, with no threshold to argue about.
    base_text = {r["depth"]: r["text"] for r in rows if r["arm"] == args.arms[0]}
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        same = sum(1 for r in sub if base_text.get(r["depth"]) == r["text"])
        mb = sum(r["pool_mb"] for r in sub) / len(sub)
        pf = sum(r["prefill_s"] for r in sub) / len(sub)
        tp = sum(r["tps"] for r in sub) / len(sub)
        rc = sum(r["recall"] for r in sub)
        rf = sum(r["res_fill"] for r in sub) / len(sub)
        bs = [r["basis"] for r in sub if r["basis"].get("enabled")]
        sh = (sum(b.get("sharing_factor", 0) for b in bs) / len(bs)) if bs else 0.0
        kp = (sum(b.get("mean_kept", 1.0) for b in bs) / len(bs)) if bs else 1.0
        if base is None:
            base = mb
        print(f"{arm:>14} {mb:>9.1f} {100*(mb-base)/max(base,1e-9):>+7.1f}% "
              f"{pf:>10.2f} {tp:>7.1f} {rc:>4}/{len(sub):<2} {rf:>8.1f} "
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
