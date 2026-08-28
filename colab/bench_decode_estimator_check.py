#!/usr/bin/env python3
"""How much can the SUBTRACTION estimator for decode rate actually resolve?

WHY THIS EXISTS
---------------
`HANDOFF_CUDA_PREFILL.md` §8 records "decode is ~12% slower after the §0.5
prefill fixes and it is NOT explained", from

    decode_tok_s = (GEN - 1) / (total_s - ttft_s)

where `total_s` and `ttft_s` are the walls of TWO SEPARATE generate() calls, each
of which runs its own full prefill (benchmarks/clean_sweep_v2.py:100-128). The
handoff already distrusts this method at small N -- it calls the -28% at
`max_new_tokens=32` "an artifact of the method; do not quote it" -- but attributes
that to fixed per-call overhead and keeps the N=128 number.

The mechanism is worse than fixed overhead. The subtraction cancels the two
prefills only to the extent they take the SAME time, and they are separate runs:
whatever spread a ~5 s prefill has lands, undivided, in an estimate of ~0.5 s of
decode. A 2% wobble in prefill is a ~20% wobble in the answer.

So this measures the estimator against itself: same build, same prompt, nothing
changed between repetitions. Any spread it reports is pure method noise, and any
build-to-build difference smaller than that spread was never resolvable.

It reports, side by side:

  estimator     (GEN-1) / (total_s - ttft_s), exactly as clean_sweep_v2 computes it
  ground truth  per-token time from DKV_TIME_ATTN's own `total_token=` traces,
                which is what colab/bench_decode_paired.py uses and which needs no
                subtraction at all

USAGE
    MODEL=Qwen/Qwen3.5-2B GEN=128 REPS=6 python colab/bench_decode_estimator_check.py
"""
import io
import os
import re
import statistics
import sys
import time
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")

from serving.decode_config import BEST_DECODE_DEFAULTS  # noqa: E402

for _k, _v in BEST_DECODE_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
os.environ["DKV_TIME_ATTN"] = "1"

import torch  # noqa: E402
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
GEN = int(os.environ.get("GEN", "128"))
REPS = int(os.environ.get("REPS", "6"))
REP = int(os.environ.get("REP", "700"))
_TOK = re.compile(r"total_token=([0-9.]+)ms")


def main():
    w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16"}, device="cuda")
    w.ensure_loaded()
    tok = w.tokenizer
    ctx = "The archive records a long sequence of unremarkable events. " * REP
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": ctx + "\nWrite a detailed essay about archives."}],
        tokenize=False, add_generation_prompt=True)
    ntok = len(tok(prompt).input_ids)
    print(f"MODEL={MODEL} CTX={ntok} GEN={GEN} REPS={REPS}", flush=True)

    def clear():
        for sid in list(getattr(w.manager, "decode_workspace", {}) or {}):
            try:
                w.manager.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass

    def timed(n, sid):
        # A FRESH session id every call, so no prefill is reused -- which is what
        # clean_sweep_v2 does and is the whole reason the two walls each carry a
        # full prefill.
        clear()
        w.active_session = sid
        buf = io.StringIO()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with redirect_stdout(buf):
            w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0,
                       top_p=1.0, repetition_penalty=1.0)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        per_tok = [float(m) for m in _TOK.findall(buf.getvalue())]
        return wall, per_tok[1:] or per_tok

    # Warm up so clock ramp and Triton autotune are not charged to rep 0.
    for _ in range(2):
        timed(8, "warm")

    rows = []
    for r in range(REPS):
        ttft_s, _ = timed(1, f"ttft-{r}")
        total_s, per_tok = timed(GEN, f"full-{r}")
        est = (GEN - 1) / (total_s - ttft_s) if total_s > ttft_s else float("nan")
        truth_ms = min(per_tok) if per_tok else float("nan")
        rows.append((ttft_s, total_s, est, truth_ms))
        print(f"  rep {r}: ttft={ttft_s:7.3f}s total={total_s:7.3f}s "
              f"-> estimator {est:7.2f} tok/s | truth {1000 / truth_ms:6.2f} tok/s "
              f"({truth_ms:.2f} ms/tok)", flush=True)

    ttfts = [r[0] for r in rows]
    ests = [r[2] for r in rows]
    truths = [1000 / r[3] for r in rows]

    def spread(xs):
        m = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, sd, 100 * sd / m, 100 * (max(xs) - min(xs)) / m

    tm, tsd, tcv, trange = spread(ttfts)
    em, esd, ecv, erange = spread(ests)
    gm, gsd, gcv, grange = spread(truths)
    print("")
    print(f"prefill wall   mean {tm:7.3f} s   sd {tsd:.3f}  cv {tcv:5.2f}%  "
          f"range {trange:5.1f}%")
    print(f"ESTIMATOR      mean {em:7.2f} tok/s  sd {esd:.2f}  cv {ecv:5.2f}%  "
          f"range {erange:5.1f}%")
    print(f"ground truth   mean {gm:7.2f} tok/s  sd {gsd:.2f}  cv {gcv:5.2f}%  "
          f"range {grange:5.1f}%")
    print("")
    decode_s = (GEN - 1) / gm
    print(f"decode being estimated: ~{decode_s:.2f} s against a prefill of "
          f"~{tm:.2f} s ({tm / decode_s:.1f}x larger)")
    print(f"a 1% wobble in prefill moves the estimator by "
          f"~{100 * 0.01 * tm / decode_s:.1f}%")
    print("")
    print("READING: any build-to-build decode difference SMALLER than the")
    print("estimator's own range above was never resolvable by this method.")

    # ── An INDEPENDENT bound on any decode rate on this card ────────────────
    # Autoregressive decode streams every weight once per token, so
    # tok/s <= bandwidth / weight_bytes no matter what the implementation does.
    # A reported rate above this line is not a fast implementation, it is a
    # broken measurement -- which is the cheapest possible check on a throughput
    # number and costs one nvidia-smi call.
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        name = q.stdout.strip().splitlines()[0]
    except Exception:                                            # noqa: BLE001
        name = "unknown GPU"
    nparam = sum(p_.numel() for p_ in w.model.parameters())
    wbytes = sum(p_.numel() * p_.element_size() for p_ in w.model.parameters())
    bw = float(os.environ.get("GPU_BW_GBPS", "504"))       # RTX 4070 SUPER spec
    ceil_tps = bw * 1e9 / wbytes
    print("")
    print(f"weights {nparam / 1e9:.2f}B params = {wbytes / 1e9:.2f} GB; "
          f"{name}, {bw:.0f} GB/s")
    print(f"BANDWIDTH CEILING on decode: {ceil_tps:.1f} tok/s "
          f"({1000 / ceil_tps:.2f} ms/tok floor)")
    print(f"  ground truth {gm:.1f} tok/s is {100 * gm / ceil_tps:.0f}% of it "
          f"-- {'plausible' if gm <= ceil_tps else 'IMPOSSIBLE'}")
    print(f"  estimator    {em:.1f} tok/s is {100 * em / ceil_tps:.0f}% of it "
          f"-- {'plausible' if em <= ceil_tps else 'IMPOSSIBLE'}")


if __name__ == "__main__":
    main()
