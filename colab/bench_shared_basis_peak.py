#!/usr/bin/env python3
"""§2b(d): does the shared-basis pool saving reach PEAK, or only the pool?

WHY THIS EXISTS
---------------
The README quotes shared bases at "pool 91.4 -> 69.8 MB (-23.6%)" and frames it
as a CAPACITY result -- `_bytes_per_block` amortises V, so the same budget holds
proportionally more blocks. That is a defensible claim and it is NOT a claim
about peak memory. It only becomes one if someone quotes it as a memory saving,
and `CUDA_TODO.md` §2b(d) asked for a peak measurement before anyone does.

On MLX the same feature halved the V store exactly as designed and moved PEAK by
1.1% at 8k and 3.4% at 32k, because peak is dominated by weights and prefill
activations rather than by the pool.

BOTH NUMBERS ARE MEASURED, IN THE SAME PROCESS, and that is the point: putting a
SYNTHETIC pool size next to a REAL peak is what produced the wrong conclusion on
the MLX side, and it took a second measurement to catch. Pool bytes here are
summed from the pool's actual allocated tensors, and peak is
torch.cuda.max_memory_allocated() around the same run.

Sharing requires an UNROTATED pool -- the pool refuses the rotated combination --
which the default preset already gives.

USAGE
    python colab/bench_shared_basis_peak.py            # both arms, 8k and 32k
    CTXS=8192 python colab/bench_shared_basis_peak.py  # one context
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")


def _pool_bytes(pool):
    """Sum the pool's REAL allocated tensors. Not _bytes_per_block * n_blocks --
    that is the budgeting formula, and the whole point of §2b(d) is not to put a
    computed number next to a measured one."""
    import torch
    seen, total = set(), 0
    for name in ("U", "U_scale", "V_KV", "anchors_KV", "scales", "seq_lens",
                 "desc", "residual_K_positions", "residual_K_values",
                 "residual_V_positions", "residual_V_values", "basis_of",
                 "U_sem", "U_sem_scale", "U_fact", "n_semantic",
                 "fact_anchors_K", "fact_anchors_V", "fact_anchor_positions"):
        t = getattr(pool, name, None)
        if torch.is_tensor(t) and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


def run_arm(share, ctx):
    import torch
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    os.chdir(ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt

    w = DKVHFWrapper(model_id=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                     config={"preset": os.environ.get("PRESET", "mid")})
    w.ensure_loaded()
    torch.cuda.synchronize()
    weights_b = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    w.active_session = f"peak-{share}-{ctx}"
    w.generate(build_prompt(w.tokenizer, ctx, 0.5), max_new_tokens=8,
               temperature=0.0, top_p=1.0, repetition_penalty=1.0,
               query_text="What is the secret passcode? Repeat it exactly.")
    torch.cuda.synchronize()
    peak_b = torch.cuda.max_memory_allocated()
    pool = getattr(w.manager, "native_pool", None)
    out = {
        "share": share, "ctx": ctx,
        "pool_mb": _pool_bytes(pool) / 1e6 if pool is not None else -1.0,
        "peak_mb": peak_b / 1e6,
        "weights_mb": weights_b / 1e6,
        "sharing_active": bool(getattr(pool, "shared_basis_active", False)),
        "v_rows": int(pool.V_KV.shape[0]) if pool is not None else -1,
        "slots": int(pool.current_blocks) if pool is not None else -1,
    }
    print("JSON " + json.dumps(out), flush=True)


def main():
    if os.environ.get("_ARM"):
        return run_arm(os.environ["_ARM"] == "on", int(os.environ["_CTX"]))
    ctxs = [int(x) for x in os.environ.get("CTXS", "8192,32768").split(",")]
    res = []
    for ctx in ctxs:
        for share in ("off", "on"):
            env = dict(os.environ, _ARM=share, _CTX=str(ctx),
                       DKV_SHARED_BASIS=("1" if share == "on" else "0"),
                       DKV_SHARED_BASIS_FRAC="0.50",
                       DKV_POOL_BUDGET_GB="2.0", DKV_RSVD_SEED="1234",
                       DKV_SVD_SEED="1234", PYTHONIOENCODING="utf-8",
                       PYTHONUTF8="1")
            p = subprocess.run([sys.executable, os.path.abspath(__file__)],
                               cwd=REPO, env=env, capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("JSON ")]
            if not line:
                print(f"ctx={ctx} share={share}: FAILED\n{p.stdout[-800:]}"
                      f"\n{p.stderr[-800:]}")
                continue
            res.append(json.loads(line[-1][5:]))
            r = res[-1]
            print(f"  ctx={ctx:>6} share={share:>3} active={r['sharing_active']!s:>5} "
                  f"v_rows={r['v_rows']:>5} slots={r['slots']:>5} "
                  f"pool={r['pool_mb']:8.1f} MB  peak={r['peak_mb']:9.1f} MB",
                  flush=True)

    print("")
    print(f"{'ctx':>7} {'pool off':>10} {'pool on':>10} {'pool D':>9} "
          f"{'peak off':>10} {'peak on':>10} {'peak D':>9}")
    for ctx in ctxs:
        a = next((r for r in res if r["ctx"] == ctx and not r["share"]), None)
        b = next((r for r in res if r["ctx"] == ctx and r["share"]), None)
        if not a or not b:
            continue
        dp = 100 * (b["pool_mb"] - a["pool_mb"]) / a["pool_mb"]
        dk = 100 * (b["peak_mb"] - a["peak_mb"]) / a["peak_mb"]
        print(f"{ctx:>7} {a['pool_mb']:10.1f} {b['pool_mb']:10.1f} {dp:+8.1f}% "
              f"{a['peak_mb']:10.1f} {b['peak_mb']:10.1f} {dk:+8.1f}%")
    if res:
        print("")
        print(f"weights alone: {res[0]['weights_mb']:.0f} MB of every peak above.")
        print("READ THE PEAK COLUMN. The pool column is a capacity result --")
        print("_bytes_per_block amortises V, so the same budget holds more")
        print("blocks. It is only a MEMORY saving to the extent peak moves.")


if __name__ == "__main__":
    main()
