#!/usr/bin/env python3
"""Low-variance paired PREFILL benchmark.

`bench_decode_paired.py` sizes decode changes. Nothing sized PREFILL, which is
why HANDOFF §8's throughput item could not be answered: the only number ever
quoted was a wall clock across two code versions (47.0s vs 48.7s) whose outputs
DIFFER once one of them starts answering correctly, so it compared two different
amounts of generated text and was never a throughput result.

That specific comparison is no longer possible -- both §0.5 fixes are permanent
and the pre-fix HEAD is gone. What was actually missing is an instrument, so
that the NEXT prefill change can be sized without repeating the mistake.

SAME DESIGN AS THE DECODE HARNESS, for the same reasons:

  * ONE process, ONE model load, ONE prompt, for both arms.
  * Arms INTERLEAVED, order alternating every round, so a clock ramp or thermal
    drift lands on both arms roughly equally.
  * PAIRED statistic: per round, stat(A) - stat(B). Drift common to a round
    cancels, so the spread of the DIFFERENCES sets the resolution, not the
    spread of the raw values.
  * MINIMUM, not median, as the per-round estimator. Timing noise is one-sided:
    a prefill can be delayed by a clock dip or the compression thread, but
    nothing makes the work finish faster than it is.

WHAT IS TIMED. `generate(max_new_tokens=1)` end to end. At >=8k the prefill
dominates a single decode step by orders of magnitude, but this is NOT a pure
prefill number and is not labelled as one -- it is "prefill plus one step",
which is the right quantity for "did my prefill change make the user wait
longer".

VALIDATE BEFORE BELIEVING AN A/B. Run MODE=AA first and confirm the CI contains
zero. An A/B whose harness has not been shown to report "no effect" when
nothing changed is not evidence.

    MODE=AA ROUNDS=6 python colab/bench_prefill_paired.py            # control
    MODE=AB EXPERIMENT=sparse_prefill ROUNDS=8 python colab/bench_prefill_paired.py
"""

import io
import os
import statistics
import sys
import time
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

os.environ.setdefault("DKV_ROUTE_PROBE", "0")
os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")
# Pin the seed-sensitive SVD draw: it does not change TIMING, but an unpinned
# rank schedule changes how much work compression does, which does.
os.environ.setdefault("DKV_RSVD_SEED", "1234")

import torch  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CTX = int(os.environ.get("CTX", "8192"))
ROUNDS = int(os.environ.get("ROUNDS", "8"))
MODE = os.environ.get("MODE", "AA")
EXPERIMENT = os.environ.get("EXPERIMENT", "sparse_prefill")
QUESTION = "What is the secret passcode? Repeat it exactly."


def set_arm(arm):
    """Apply an arm's config. A is the 'new'/'on' side by convention.

    Only knobs read from the environment at CALL time belong here. A knob
    captured into a module constant at import must be rebound on the module
    instead -- bench_decode_paired.py records two arms (`remat`, `fastdc`)
    where that distinction was missed and the harness truthfully reported "no
    effect" for a change it never made.
    """
    if MODE == "AA":
        return
    on = (arm == "A")
    if EXPERIMENT == "sparse_prefill":
        os.environ["DKV_SPARSE_PREFILL"] = "1" if on else "0"
    elif EXPERIMENT == "shared_basis":
        # Compression-time cost of shared bases: the registry scores each block
        # against candidate groups and re-projects U.
        os.environ["DKV_SHARED_BASIS"] = "1" if on else "0"
        os.environ.setdefault("DKV_SHARED_BASIS_FRAC", "0.50")
    elif EXPERIMENT == "gram_svd":
        os.environ["DKV_COMPRESS_GRAM_SVD"] = "1" if on else "0"
    elif EXPERIMENT == "rotated_pool":
        os.environ["DKV_ROTATED_POOL"] = "0" if on else "1"
    else:
        raise SystemExit(f"unknown EXPERIMENT={EXPERIMENT}")


def clocks():
    try:
        import subprocess
        q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        sm, t = q.stdout.strip().splitlines()[0].split(",")
        return int(sm), int(t)
    except Exception:                                            # noqa: BLE001
        return (-1, -1)


def main():
    os.chdir(os.path.join(ROOT, "ACTIVE_RUNTIME"))
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt

    w = DKVHFWrapper(model_id=MODEL,
                     config={"quantization": None, "rank": 32, "block_size": 256,
                             "micro_block_size": 256, "preset": "mid"})
    w.ensure_loaded()
    prompt = build_prompt(w.tokenizer, CTX, 0.5)
    n_tok = len(w.tokenizer.encode(prompt))
    print(f"CTX {n_tok} mode={MODE} exp={EXPERIMENT} rounds={ROUNDS}", flush=True)

    def run(sid):
        # Clear first: each generate() re-prefills and leaves session and pool
        # state behind, and letting that accumulate across the ~16 calls of a
        # paired run makes later rounds slower AND noisier -- which shows up as
        # a systematic bias in the A/A control even with both arms identical.
        try:
            w.clear_session(sid)
        except Exception:                                        # noqa: BLE001
            pass
        w.active_session = sid
        torch.cuda.synchronize()
        t = time.perf_counter()
        buf = io.StringIO()
        with redirect_stdout(buf):
            w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0, query_text=QUESTION)
        torch.cuda.synchronize()
        el = time.perf_counter() - t
        try:
            w.clear_session(sid)
        except Exception:                                        # noqa: BLE001
            pass
        return el * 1000.0

    c0 = clocks()
    for _ in range(2):
        run("warm")
    torch.cuda.synchronize()
    c1 = clocks()
    print(f"CLOCKS idle={c0[0]}MHz/{c0[1]}C warm={c1[0]}MHz/{c1[1]}C", flush=True)

    per_round, a_all, b_all = [], [], []
    for r in range(ROUNDS):
        order = ("A", "B") if r % 2 == 0 else ("B", "A")
        res = {}
        for arm in order:
            set_arm(arm)
            res[arm] = min(run(f"pf-{arm}") for _ in range(2))
        per_round.append(res["A"] - res["B"])
        a_all.append(res["A"])
        b_all.append(res["B"])
        print(f"  round {r}: A={res['A']:.1f} B={res['B']:.1f} "
              f"diff={res['A'] - res['B']:+.1f} ms  (order {order[0]} first)",
              flush=True)

    n = len(per_round)
    mean_d = statistics.mean(per_round)
    sd_d = statistics.stdev(per_round) if n > 1 else 0.0
    tcrit = 2.36 if n <= 8 else 2.0
    half = tcrit * sd_d / (n ** 0.5) if n > 1 else float("inf")
    med_a, med_b = statistics.median(a_all), statistics.median(b_all)
    print(f"RESULT A={med_a:.1f} ms  B={med_b:.1f} ms  "
          f"({n_tok / (med_b / 1000.0):.0f} tok/s prefill on B)", flush=True)
    print(f"PAIRED mean_diff={mean_d:+.2f} ms  95%CI=[{mean_d - half:+.2f}, "
          f"{mean_d + half:+.2f}]  resolution=+-{half:.2f} ms "
          f"({100 * half / med_b:.1f}% of a prefill)", flush=True)
    verdict = ("NO EFFECT RESOLVABLE (CI contains 0)"
               if (mean_d - half) * (mean_d + half) <= 0
               else f"EFFECT: A is {'slower' if mean_d > 0 else 'faster'} "
                    f"by {abs(100 * mean_d / med_b):.1f}%")
    print(f"VERDICT {verdict}", flush=True)


if __name__ == "__main__":
    main()
