"""Low-variance paired decode benchmark.

Comparing two configs by running two processes and reading the medians cannot
resolve anything below ~20% here: the same config measured 12.14 and 10.15 tok/s
minutes apart. The GPU idles at 210 MHz against a 3120 MHz boost ceiling and the
clocks cannot be pinned without admin, so a run's absolute speed depends on where
the clock and thermal state happened to be.

The fix is to never compare across processes:

  * ONE process, ONE model load, ONE context, for both arms.
  * Config is flipped by rebinding the module-level constant, which the patched
    forward resolves from module globals at call time, so both arms share
    everything else including the allocator's state.
  * Arms are INTERLEAVED and their order alternates every round, so a clock ramp
    or thermal drift lands on both arms roughly equally instead of on whichever
    ran first.
  * The statistic is PAIRED: per round, stat(A) - stat(B). Drift common to a
    round cancels in the difference, so the spread of those differences -- not
    the spread of the raw values -- sets the resolution.

Calibration on Qwen2.5-1.5B (see stat() and run() for why min and why the session
is cleared):

    A/A control  8 rounds @8.4k   mean_diff +0.135 ms, CI [-1.277, +1.547]
                                  -> correctly reports no effect
    A/B known    8 rounds @8.4k   mean_diff -2.151 ms, CI [-2.884, -1.419]
                                  -> detects DKV_GRAPH_SAFE_ROUTING, 8/8 rounds
    A/B known   12 rounds @32k    mean_diff -4.072 ms, CI [-6.006, -2.137]

Resolution is ~+-1.8% of a token at 8.4k and ~+-4% at 32k (the longer context is
noisier: bigger pool, compression running during decode). Before the pairing and
the min estimator the same comparison could not resolve 20%.

USE THIS, NOT cProfile SELF-TIME, TO SIZE A CHANGE. cProfile charges its own
per-call overhead to the callee, so a high-count cheap builtin looks expensive:
its decode attribution put .to() at 6.21 ms/token over 365 calls, but cutting
those calls 229.8 -> 56.4 (a manager-owned cache for the key boxes, which worked
mechanically) produced NO resolvable change here. Tensor.to() returns self when
the tensor is already on the right device, so it costs dispatch and nothing else.
cProfile is a good tool for finding WHICH code issues the calls; it is a bad tool
for deciding whether removing them is worth anything.

Usage:
    MODE=AA  ROUNDS=8  REP=700  python colab/bench_decode_paired.py   # control
    MODE=AB  ROUNDS=12 REP=2800 python colab/bench_decode_paired.py   # comparison

Point set_arm() at whatever module constant you are testing.

Reports the paired mean difference with a 95% CI. If the CI straddles zero the
harness is saying "no effect resolvable", which is a real answer.

Validate the harness before believing an A/B: run it in A/A mode (both arms the
same) and confirm the CI contains zero, then on a known effect and confirm it
does not.
"""
import io
import os
import re
import statistics
import sys
import time
from contextlib import redirect_stdout

import torch

ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS  # noqa: E402

for _k, _v in BEST_DECODE_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
os.environ["DKV_TIME_ATTN"] = "1"
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper  # noqa: E402
import runtime.dkv_attention as DA  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
REP = int(os.environ.get("REP", "700"))
ROUNDS = int(os.environ.get("ROUNDS", "8"))
NTOK = int(os.environ.get("NTOK", "24"))
MODE = os.environ.get("MODE", "AA")          # AA = control, AB = real comparison
_TOK = re.compile(r"total_token=([0-9.]+)ms")


# EXPERIMENT selects what the two arms differ by. Knobs read from os.environ at
# CALL time can simply be set here; knobs captured into a module constant at
# import time must be rebound on the module, which is why both forms appear.
EXPERIMENT = os.environ.get("EXPERIMENT", "routing")


def set_arm(arm):
    """Apply an arm's config. A is the 'new'/'on' side by convention."""
    if MODE == "AA":
        return                                # both arms identical: the control
    on = (arm == "A")
    if EXPERIMENT == "routing":
        DA._GRAPH_SAFE_ROUTING = on
    elif EXPERIMENT == "topk":
        # Read at call time by query_router, so the environment is enough.
        if on:
            os.environ["DKV_TOPK_BLOCKS"] = os.environ.get("TOPK_ON", "32")
        else:
            os.environ.pop("DKV_TOPK_BLOCKS", None)
    elif EXPERIMENT == "topk_pair":
        # BOTH arms explicit, so any two K values can be compared. `topk` above
        # pins the off-arm to whatever the pool default happens to be, which
        # makes it useless once that default is the value under test -- and it
        # silently changes meaning when the default changes.
        # TOPK_OFF=0 means "attend every block", the widest contrast available.
        os.environ["DKV_TOPK_BLOCKS"] = os.environ.get(
            "TOPK_ON" if on else "TOPK_OFF", "4" if on else "0")
    elif EXPERIMENT == "remat_interval":
        os.environ["DKV_REMAT_INTERVAL"] = (os.environ.get("IV_ON", "16") if on
                                            else os.environ.get("IV_OFF", "4"))
    elif EXPERIMENT == "remat":
        # NOT the environment: dkv_attention captures remat_enabled() into
        # _REMAT_ENABLED at import, so setting DKV_REMAT_CACHE here would leave
        # both arms on and the benchmark would truthfully report "no effect" for
        # a change it never actually made.
        #
        # Run with DKV_SPARSE_BIAS=0.0 exported to price remat on the COMBINED
        # branch (the library default); BEST_DECODE_DEFAULTS' setdefault
        # otherwise puts it on "auto" and measures the production branch.
        # Measured on the combined branch at 8.4k, Qwen2.5-1.5B, 8 rounds:
        # A=54.75 ms/tok B=78.43, mean_diff -23.459 ms, CI [-24.041, -22.876],
        # i.e. remat is worth 29.9% there. That number is what made passing it a
        # correctly-framed dense window the right fix rather than declining.
        DA._REMAT_ENABLED = on
    elif EXPERIMENT == "dense_ring":
        # Requires the append-only patch in assemble_dense_window_kv, which is
        # NOT in the tree: measured with this harness it was neutral at both
        # 8.4k (CI [-1.42, +0.83] ms) and 32k (CI [-1.53, +2.82] ms), so it was
        # dropped rather than carried for no gain. Re-apply the patch to use it.
        import native_core.kv_runtime_manager as KVM
        if not hasattr(KVM, "_DENSE_RING"):
            raise SystemExit("dense_ring: append-only patch not present in tree")
        KVM._DENSE_RING = on
    elif EXPERIMENT in ("router_rope", "residual_exact_rope"):
        # WHERE DOES THE UNROTATED POOL SPEND ITS 43-137%? Both of these run ONLY
        # when the pool is unrotated, so run them with DKV_ROTATED_POOL=0. A is
        # the CHEAP side (feature off), so a negative mean_diff is what that
        # feature costs.
        #
        #   router_rope         the router rotates anchors before scoring, which
        #                       a rotated pool skips entirely (it would be a
        #                       second rotation).
        #   residual_exact_rope rotates each exact residual at its TRUE position
        #                       rather than the block anchor, growing the cos/sin
        #                       gather from [N, D] to [N, MAX_RES, D].
        #
        # Both are read from the environment at call time, so setting them here
        # is enough -- no module rebinding needed.
        _var = ("DKV_ROUTER_ROPE" if EXPERIMENT == "router_rope"
                else "DKV_RESIDUAL_EXACT_ROPE")
        os.environ[_var] = "0" if on else "1"
    elif EXPERIMENT == "exact_rope_remat":
        # Cost of rotating each MATERIALISED key at its own absolute position
        # instead of rotating the basis once at the block anchor. Paid inside the
        # RematCache entry, so once per refresh rather than once per token.
        os.environ["DKV_EXACT_ROPE_REMAT"] = "1" if on else "0"
    elif EXPERIMENT == "router_exact_key":
        # Cost of folding the anchor into the residual key before rotating it:
        # one broadcast add on [N, R, H_kv, D] per routing call, alongside the
        # rotation that already runs on the same tensor.
        os.environ["DKV_ROUTER_EXACT_KEY"] = "1" if on else "0"
    elif EXPERIMENT == "rotated_pool":
        # A = UNROTATED (the candidate default), B = rotated (the historical one),
        # so a POSITIVE mean_diff is what unrotated COSTS.
        #
        # Safe to flip in-process only because both sides read the flag at call
        # time -- _ingest_k() picks which K prefill STORES and the decode gather
        # picks whether to re-rotate, both via pool_stores_rotated_k(). run()
        # clears the session and re-prefills on every call, so each arm stores
        # and reads consistently. If either side had captured the flag at import
        # this would store one way and read the other, which is not a slow
        # config -- it is RoPE missing from all compressed content, and it would
        # show up as a timing difference rather than as the garbage it is.
        os.environ["DKV_ROTATED_POOL"] = "0" if on else "1"
    elif EXPERIMENT == "fastdc":
        # --fastdc could not be A/B'd in one process before, which is why its
        # worth was argued from cross-process walls that could not resolve it.
        # Three things have to move together, and each for a different reason:
        #
        #   DKV_FAST_DECODE / DKV_GRAPH_FORCE_ROUTED are re-read from the
        #   environment on every decode step, so setting them here is enough.
        #
        #   _MUTATION_OUT is captured into a module constant at import. The
        #   wrapper's _dkv_publish_mutation_out reads it each step to decide
        #   _MUTATION_OUT_ACTIVE, so rebinding it here really does switch the
        #   forward's behaviour -- setting only the environment would leave both
        #   arms identical and the harness would truthfully report "no effect"
        #   for a change it never made (the same trap as the `remat` arm).
        #
        #   The CAPTURED GRAPH must not outlive the switch. Replay executes no
        #   Python, so replaying arm A's graph while arm B expects a mutating
        #   forward advances no state -- wrong text and a meaningless time.
        on_ = "1" if on else "0"
        os.environ["DKV_FAST_DECODE"] = on_
        os.environ["DKV_GRAPH_FORCE_ROUTED"] = on_
        DA._MUTATION_OUT = on
        _w = globals().get("_WRAPPER")
        if _w is not None and getattr(_w, "_cuda_graph_runner", None) is not None:
            _w._cuda_graph_runner.invalidate()
            _w._dkv_capture_giveup = False
    elif EXPERIMENT == "graph_safe_routed":
        # DKV_GRAPH_SAFE_DECODE=1 was shipped after an accuracy check but no
        # speed check. On the routed path it forces `changed = True` every step,
        # which discards the gather cache on every token. A is the RELAXED side
        # (constant False = exact comparison restored), B is the shipped-broken
        # side, so a NEGATIVE mean_diff means the fix is faster.
        DA._GRAPH_SAFE_ROUTED = not on
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
    except Exception:
        return -1, -1


def main():
    w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16"}, device="cuda")
    w.ensure_loaded()
    globals()["_WRAPPER"] = w      # set_arm needs it to invalidate the graph
    tok = w.tokenizer
    ctx = "The archive records a long sequence of unremarkable events. " * REP
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": ctx + "\nWrite a detailed essay about archives."}],
        tokenize=False, add_generation_prompt=True)
    print(f"CTX {len(tok(prompt).input_ids)} mode={MODE} exp={EXPERIMENT} rounds={ROUNDS} ntok={NTOK}",
          flush=True)

    def run(n):
        # Clear the session first. Each generate() re-prefills and leaves session
        # and pool state behind; letting that accumulate across the ~16 calls of a
        # paired run made later rounds both slower and noisier, which showed up as
        # a systematic bias in the A/A control even though both arms were the same
        # config.
        for _sid in list(getattr(w.manager, "decode_workspace", {}) or {}):
            try:
                w.manager.clear_session(_sid)
            except Exception:
                pass
        buf = io.StringIO()
        with redirect_stdout(buf):
            w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0)
        s = [float(m) for m in _TOK.findall(buf.getvalue())]
        return s[1:] or s          # drop the first: it carries the prefill transition

    def stat(samples):
        # Minimum, not median. Timing noise here is ONE-SIDED -- a token can be
        # delayed by a clock dip, an allocator hiccup or the compression thread,
        # but nothing can make the work finish faster than it actually is. The
        # minimum over the samples is therefore the cleanest estimate of the true
        # per-token cost, and it ignores exactly the spikes that were dominating
        # the median.
        return min(samples)

    # Sustained warm-up: bring the clocks off idle and let Triton autotune settle
    # BEFORE any measurement, so the ramp is not charged to whichever arm is first.
    c0 = clocks()
    for _ in range(3):
        run(NTOK)
    torch.cuda.synchronize()
    c1 = clocks()
    print(f"CLOCKS idle={c0[0]}MHz/{c0[1]}C warm={c1[0]}MHz/{c1[1]}C", flush=True)

    per_round = []
    a_all, b_all = [], []
    for r in range(ROUNDS):
        # Alternate which arm leads, so an intra-round ramp does not always
        # favour the same one.
        order = ("A", "B") if r % 2 == 0 else ("B", "A")
        res = {}
        for arm in order:
            set_arm(arm)
            res[arm] = stat(run(NTOK))
        per_round.append(res["A"] - res["B"])
        a_all.append(res["A"])
        b_all.append(res["B"])
        print(f"  round {r}: A={res['A']:.2f} B={res['B']:.2f} "
              f"diff={res['A'] - res['B']:+.2f} ms  (order {order[0]} first)",
              flush=True)

    n = len(per_round)
    mean_d = statistics.mean(per_round)
    sd_d = statistics.stdev(per_round) if n > 1 else 0.0
    # 95% CI on the paired mean (t ~ 2.36 at 7 dof; 2.0 is close enough above ~10)
    tcrit = 2.36 if n <= 8 else 2.0
    half = tcrit * sd_d / (n ** 0.5) if n > 1 else float("inf")
    med_a, med_b = statistics.median(a_all), statistics.median(b_all)
    cv_a = 100 * statistics.stdev(a_all) / med_a if n > 1 else 0.0
    cv_b = 100 * statistics.stdev(b_all) / med_b if n > 1 else 0.0
    print(f"RESULT A={med_a:.2f} ms/tok ({1000 / med_a:.2f} tok/s, cv={cv_a:.1f}%)  "
          f"B={med_b:.2f} ms/tok ({1000 / med_b:.2f} tok/s, cv={cv_b:.1f}%)", flush=True)
    print(f"PAIRED mean_diff={mean_d:+.3f} ms  95%CI=[{mean_d - half:+.3f}, "
          f"{mean_d + half:+.3f}]  resolution=+-{half:.3f} ms "
          f"({100 * half / med_b:.1f}% of a token)", flush=True)
    verdict = ("NO EFFECT RESOLVABLE (CI contains 0)"
               if (mean_d - half) * (mean_d + half) <= 0
               else f"EFFECT: A is {'slower' if mean_d > 0 else 'faster'} "
                    f"by {abs(100 * mean_d / med_b):.1f}%")
    print(f"VERDICT {verdict}", flush=True)


if __name__ == "__main__":
    main()
