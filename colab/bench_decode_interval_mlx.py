#!/usr/bin/env python3
"""Paired decode benchmark for MLX — built to settle DKV_DECODE_CACHE_INTERVAL.

WHY PAIRED, AND WHY NOT TWO PROCESSES
-------------------------------------
Decode throughput on this machine drifts with clock and thermal state, by more
than most of the effects worth measuring. Comparing two runs' medians cannot
resolve anything small. So:

  * ONE process, ONE model load, ONE prefill, for both arms.
  * The arm is flipped by rebinding manager._decode_cache_interval, which the
    decode path reads per step — so both arms share everything else, including
    the pool and the allocator's state.
  * Arms are INTERLEAVED and their order ALTERNATES every round, so a clock ramp
    lands on both roughly equally instead of on whichever ran first.
  * The statistic is PAIRED: per round, A - B. Drift common to a round cancels,
    so the spread of the DIFFERENCES sets the resolution, not the spread of the
    raw numbers.
  * MIN, not median, per round: timing noise here is one-sided (something else
    stealing the GPU can only make a step slower).
  * The session is ROLLED BACK to the prefill length between arms, so each arm
    decodes from an identical state rather than inheriting the other's tokens.

CALIBRATE BEFORE BELIEVING IT: `MODE=AA` runs the same interval in both arms and
must report no effect. A harness that cannot pass its own A/A control cannot be
trusted to report an A/B.

    MODE=AA ROUNDS=6 python colab/bench_decode_interval_mlx.py
    MODE=AB ARM_A=16 ARM_B=4 ROUNDS=6 python colab/bench_decode_interval_mlx.py
"""
import math
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))

import mlx.core as mx                                            # noqa: E402
import torch                                                     # noqa: E402

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3.5-2B-4bit")
MODE = os.environ.get("MODE", "AB")
ARM_A = int(os.environ.get("ARM_A", "16"))
ARM_B = int(os.environ.get("ARM_B", "4"))
ROUNDS = int(os.environ.get("ROUNDS", "6"))
STEPS = int(os.environ.get("STEPS", "48"))
NFILL = int(os.environ.get("NFILL", "700"))

FILLER = [
    "The morning fog rolled over the hills before the sun broke through the clouds.",
    "Researchers published a new dataset covering climate trends across five continents.",
    "The old library smelled of dust and aging paper, a comfort to regular visitors.",
    "Markets fluctuated throughout the week as investors weighed new economic data.",
]


def main():
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    from serving.mlx_dkv_wrapper import MLXDKVWrapper

    w = MLXDKVWrapper(model_id=MODEL, config={"preset": "mid"})
    w.ensure_loaded()
    mgr = w.manager

    random.seed(5)
    body = " ".join(random.choice(FILLER) for _ in range(NFILL))
    prompt = w.tokenizer.apply_chat_template(
        [{"role": "user", "content": body + " Summarise the passage."}],
        tokenize=False, add_generation_prompt=True)
    ids = w.tokenizer.encode(prompt)
    sid = "bench"

    # ONE prefill, reused by every measurement.
    mgr.clear_session(sid)
    mgr.init_session(sid, prefill_len=len(ids), max_tokens_hint=STEPS + 8)
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    w.model._get_or_create_prefill_cache(tuple([sid]), total_tokens=len(ids))
    w.model._dkv_session_ids = [sid]
    chunk = 1024
    for s0 in range(0, len(ids), chunk):
        piece = ids[s0:s0 + chunk]
        out = w.model(torch.tensor([piece], dtype=torch.long),
                      torch.tensor([list(range(s0, s0 + len(piece)))], dtype=torch.long))
        mgr.compress_deferred_prefill_blocks(sid)
        mx.eval(out.logits)
    base_len = len(ids)
    nb0 = mgr.sessions[sid]["num_blocks"]
    print(f"  prefilled {base_len} tokens, blocks/layer={max(nb0)}, "
          f"steps/measure={STEPS}", flush=True)

    def decode_ms_per_token(interval):
        """min ms/token over STEPS decode steps at this cache interval."""
        mgr._decode_cache_interval = max(1, interval)
        # Drop any cached route so the arm is not credited with the other's cache.
        for s in mgr.sessions.values():
            if isinstance(s, dict) and isinstance(s.get("_cache_kv"), dict):
                s["_cache_kv"].clear()
        w.model._dkv_session_ids = [sid]
        tok = ids[-1]
        best = float("inf")
        for i in range(STEPS):
            pos = base_len + i
            t0 = time.perf_counter()
            o = w.model(torch.tensor([[tok]], dtype=torch.long),
                        torch.tensor([[pos]], dtype=torch.long))
            mx.eval(o.logits)
            dt = (time.perf_counter() - t0) * 1000.0
            tok = int(o.logits[0, -1].argmax())
            if i >= 4:                      # discard warmup steps
                best = min(best, dt)
        w.rollback_session(sid, base_len)
        return best

    a_int = ARM_A
    b_int = ARM_A if MODE == "AA" else ARM_B
    print(f"  MODE={MODE}  arm A interval={a_int}  arm B interval={b_int}\n", flush=True)

    diffs, A, B = [], [], []
    for r in range(ROUNDS):
        if r % 2 == 0:                      # alternate order every round
            a = decode_ms_per_token(a_int); b = decode_ms_per_token(b_int)
        else:
            b = decode_ms_per_token(b_int); a = decode_ms_per_token(a_int)
        A.append(a); B.append(b); diffs.append(a - b)
        print(f"  round {r}: A={a:.3f} ms  B={b:.3f} ms  diff={a-b:+.3f}", flush=True)

    n = len(diffs)
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("inf")
    print(f"\n  A (interval {a_int}) mean {statistics.mean(A):.3f} ms")
    print(f"  B (interval {b_int}) mean {statistics.mean(B):.3f} ms")
    print(f"  paired mean_diff {m:+.3f} ms, 95% CI [{m-half:+.3f}, {m+half:+.3f}]")
    if m - half <= 0.0 <= m + half:
        print("  -> CI straddles zero: NO RESOLVABLE DIFFERENCE at this round count.")
    else:
        faster = "B" if m > 0 else "A"
        print(f"  -> {faster} faster by {abs(m)/statistics.mean(A)*100:.1f}%  "
              f"({sum(1 for d in diffs if (d > 0) == (m > 0))}/{n} rounds same sign)")


if __name__ == "__main__":
    main()
