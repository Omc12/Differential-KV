#!/usr/bin/env python3
"""STANDALONE Gram-eigh decision test — run this now, then decide the default.

The Gram-eigh compress path (lowrank.py, opt-in DIFFKV_COMPRESS_GRAM_SVD=1)
replaces the wide per-block SVD (cuSOLVER batched cap = 32x32 → ~2,352 sequential
decompositions/prefill) with an eigendecomposition of the small [r_proj,r_proj]
Gram matrix. This script answers, in ONE run, "where do things stand?":

  PART 1 (CPU, always runs) — CORRECTNESS
    Reproduces the exact shipped Gram-eigh vs SVD math and checks they give the
    same low-rank factorization the pipeline uses (reconstruction parity).

  PART 2 (GPU, runs if CUDA + a model) — SPEED + RECALL
    Real prefill-compress A/B on the model: baseline SVD vs Gram-eigh vs
    Gram-eigh + r_proj<=32 recipe. Reports compress time (should drop) and NIAH
    recall (must hold — the fix must not cost the 'dense-like quality' claim).

VERDICT: if PART 1 passes AND PART 2 shows recall within noise of baseline, it is
safe to make Gram-eigh the default (see the printed instructions).

Usage:
    python colab/gram_eigh_decision.py                       # CPU correctness only
    python colab/gram_eigh_decision.py --gpu-ab              # + real GPU A/B
    python colab/gram_eigh_decision.py --gpu-ab --model Qwen/Qwen2.5-7B-Instruct \\
        --ctx 16384 --samples 3
"""
import os
import sys
import argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

REAL_N, REAL_RPROJ, REAL_FEAT = 49, 37, 2048   # real prefill batch shapes
INT8_U_FLOOR = 9.2e-3                           # downstream int8-U quant error


# ── PART 1: CPU numerical parity (exact copy of lowrank.py:774-785) ──────────

def _gram_factor(B):
    G = torch.matmul(B, B.transpose(1, 2))
    evals, evecs = torch.linalg.eigh(G)
    evals = evals.flip(-1).clamp(min=0.0)
    U_b = evecs.flip(-1)
    S = evals.sqrt()
    Vh = torch.matmul(U_b.transpose(1, 2), B) / S.clamp(min=1e-8).unsqueeze(-1)
    return U_b, S, Vh


def _svd_factor(B):
    return torch.linalg.svd(B, full_matrices=False)


def _recon(U_b, S, Vh, k=None):
    if k is not None:
        U_b, S, Vh = U_b[..., :k], S[..., :k], Vh[..., :k, :]
    return torch.matmul(U_b * S.unsqueeze(-2), Vh)


def _make_B(kind, seed=0):
    g = torch.Generator().manual_seed(seed)
    N, r, f = REAL_N, REAL_RPROJ, REAL_FEAT
    if kind == "wellcond":
        return torch.randn(N, r, f, generator=g)
    if kind == "lowrank":
        A = torch.randn(N, r, 8, generator=g)
        C = torch.randn(N, 8, f, generator=g)
        return torch.matmul(A, C) + 0.01 * torch.randn(N, r, f, generator=g)
    if kind == "rankdeficient":
        A = torch.randn(N, r, 5, generator=g)
        C = torch.randn(N, 5, f, generator=g)
        return torch.matmul(A, C)
    raise ValueError(kind)


def _kept_singular_rel_error(Sg, Ss):
    worst = 0.0
    sq = Ss ** 2
    cutoff = 0.999 * sq.sum(dim=-1)
    cum = torch.cumsum(sq, dim=-1)
    for i in range(Ss.shape[0]):
        k99 = int((cum[i] < cutoff[i]).sum().item()) + 1
        rel = ((Sg[i, :k99] - Ss[i, :k99]).abs() / Ss[i, :k99].clamp(min=1e-8)).max().item()
        worst = max(worst, rel)
    return worst


def part1_cpu_correctness() -> bool:
    print("=" * 74)
    print("PART 1 — CPU numerical correctness (Gram-eigh ≡ SVD for the pipeline)")
    print("=" * 74)
    ok_all = True
    for kind in ("wellcond", "lowrank", "rankdeficient"):
        B = _make_B(kind)
        Bn = B.norm()
        Ug, Sg, Vg = _gram_factor(B)
        Us, Ss, Vs = _svd_factor(B)
        err_g = ((_recon(Ug, Sg, Vg) - B).norm() / Bn).item()
        err_s = ((_recon(Us, Ss, Vs) - B).norm() / Bn).item()
        s_rel = _kept_singular_rel_error(Sg, Ss)
        trunc_ok = True
        for k in (4, 8, 16, 32):
            eg = ((_recon(Ug, Sg, Vg, k) - B).norm() / Bn).item()
            es = ((_recon(Us, Ss, Vs, k) - B).norm() / Bn).item()
            trunc_ok = trunc_ok and abs(eg - es) < 1e-3
        ok = (err_g <= max(err_s * 1.05, 1e-4)) and (err_g < INT8_U_FLOOR) and (s_rel < 1e-3) and trunc_ok
        ok_all = ok_all and ok
        print(f"  [{kind:14}] full recon gram={err_g:.2e} svd={err_s:.2e} | kept-S rel={s_rel:.1e} "
              f"| trunc parity={'ok' if trunc_ok else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")
    # full-pipeline Q@U_b product parity
    delta = _make_B("lowrank", seed=1)
    g = torch.Generator().manual_seed(2)
    Omega = torch.randn(REAL_N, REAL_FEAT, REAL_RPROJ, generator=g)
    Y = torch.matmul(delta, Omega)
    for _ in range(2):
        Y = torch.matmul(delta, torch.matmul(delta.transpose(1, 2), Y))
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = torch.matmul(Q.transpose(1, 2), delta)
    Ug, Sg, Vg = _gram_factor(B)
    Us, Ss, Vs = _svd_factor(B)
    fg = ((torch.matmul(Q, _recon(Ug, Sg, Vg)) - delta).norm() / delta.norm()).item()
    fs = ((torch.matmul(Q, _recon(Us, Ss, Vs)) - delta).norm() / delta.norm()).item()
    pipe_ok = abs(fg - fs) < 1e-4
    ok_all = ok_all and pipe_ok
    print(f"  [full pipeline ] Q@U_b product gram={fg:.2e} svd={fs:.2e} Δ={abs(fg-fs):.1e} "
          f"-> {'PASS' if pipe_ok else 'FAIL'}")
    print(f"\n  PART 1: {'CORRECTNESS VERIFIED ✓' if ok_all else 'FAILED ✗'} "
          f"(all errors << int8-U quant floor {INT8_U_FLOOR})\n")
    return ok_all


# ── PART 2: real GPU compress A/B (speed + recall) ───────────────────────────

def part2_gpu_ab(model_id: str, ctx: int, samples: int):
    """Returns True (safe), False (not safe / inconclusive), or None (not run)."""
    print("=" * 74)
    print(f"PART 2 — GPU prefill-compress A/B @ {ctx} (compress time + NIAH recall)")
    print("=" * 74)
    if not torch.cuda.is_available():
        print("  CUDA not available — skipping GPU A/B. Run this on the A100.\n")
        return None

    # Run each sample in a fresh subprocess. The DiffKV wrapper does not fully free
    # on del, so reloading per sample OOMs a 40GB card (that is exactly what turned
    # the earlier run into all-OOM). Isolation gives every run a clean GPU.
    os.environ["DIFFKV_ISOLATE_WORKERS"] = "1"
    sys.path.insert(0, HERE)
    from run_a100_paper_experiments import run_worker_task, generate_random_needles, compute_stats, wilson_ci

    needles = generate_random_needles(samples + 2)
    recipes = [
        ("baseline_svd", {}),
        ("gram_eigh", {"DIFFKV_COMPRESS_GRAM_SVD": "1"}),
        ("gram_rproj32", {"DIFFKV_COMPRESS_GRAM_SVD": "1", "DIFFKV_RANK_BOOST": "off",
                          "DIFFKV_RSVD_MAX_RPROJ": "32", "DIFFKV_RSVD_OVERSAMPLES": "0"}),
    ]
    table = {}
    for name, env in recipes:
        comp, passes, n_ok = [], 0, 0
        for s in range(samples):
            code, sent = needles[s]
            res = run_worker_task("niah", {"model_id": model_id, "preset": "mid", "ctx_len": ctx,
                                           "depth": 0.5, "needle_code": code, "needle_sent": sent,
                                           "gen_len": 32, "extra_env": env, "n_trials": 2})
            if res.get("status") != "success":
                print(f"    [{name}] sample {s} FAILED: {res.get('error','')[:120]}")
                continue
            n_ok += 1
            comp.append(res.get("prefill_compress_s", 0.0))
            passes += int(code.upper() in res.get("output_text", "").upper())
        cs = compute_stats(comp)
        recall = (passes / n_ok * 100.0) if n_ok else 0.0
        rci = wilson_ci(passes, n_ok)["margin"] if n_ok else 0.0
        table[name] = {"compress_s": cs["mean"], "recall": recall, "recall_ci": rci,
                       "n_ok": n_ok}
        tag = "OK" if n_ok == samples else f"INCOMPLETE {n_ok}/{samples}"
        print(f"  {name:<14} [{tag}] compress={cs['mean']:.3f}s recall={recall:.0f}% (n={n_ok})")

    # ── Health gates: a decision tool must NOT say "safe" when runs crashed ──
    base = table.get("baseline_svd", {})
    print()
    if base.get("n_ok", 0) < samples or base.get("compress_s", 0.0) <= 0:
        print("  ✗ INCONCLUSIVE: baseline_svd did not complete all samples (likely OOM).")
        print("    → lower --ctx / --samples, or free the GPU, then re-run.\n")
        return False
    if base.get("recall", 0.0) <= 0:
        print("  ✗ INCONCLUSIVE: baseline_svd recall is 0% — the harness isn't retrieving the")
        print("    needle even without Gram-eigh (a separate recall/OOM problem, not compress).")
        print("    → fix retrieval first; Gram-eigh can't be judged against a broken baseline.\n")
        return False

    base_t, base_r, base_rci = base["compress_s"], base["recall"], base["recall_ci"]
    eigh_safe = True
    for name in ("gram_eigh", "gram_rproj32"):
        r = table.get(name, {})
        complete = r.get("n_ok", 0) == samples and r.get("compress_s", 0.0) > 0
        faster = complete and r["compress_s"] < base_t
        held = complete and (r["recall"] >= base_r - (r.get("recall_ci", 0.0) + base_rci))
        speed = (base_t / r["compress_s"]) if r.get("compress_s", 0.0) > 0 else 0.0
        ok = complete and faster and held
        if name == "gram_eigh":
            # Plain Gram-eigh is algebraically the SVD factorization → recall must
            # hold; speed is the whole point. Both required to call it safe.
            eigh_safe = ok
        print(f"  {name:<14} complete={complete} speedup={speed:.2f}x recall_holds={held} "
              f"-> {'SAFE' if ok else 'NOT SAFE'}")
    print(f"\n  PART 2: {'gram_eigh faster + recall held ✓' if eigh_safe else 'NOT safe — see rows above ✗'}\n")
    return eigh_safe


def main():
    ap = argparse.ArgumentParser(description="Standalone Gram-eigh decision test")
    ap.add_argument("--gpu-ab", action="store_true", help="also run the real GPU compress A/B")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    p1 = part1_cpu_correctness()
    p2 = None
    if args.gpu_ab:
        p2 = part2_gpu_ab(args.model, args.ctx, args.samples)
    else:
        print("(run with --gpu-ab on the A100 to measure compress speedup + recall)\n")

    print("=" * 74)
    if not p1:
        print("VERDICT: NOT SAFE — the Gram-eigh math failed CPU parity (Part 1). Stop here.")
        ok = False
    elif p2 is None:
        print("VERDICT: math verified (Part 1). SPEED + RECALL NOT YET MEASURED.")
        print("  Run on the A100:  python colab/gram_eigh_decision.py --gpu-ab \\")
        print("      --model Qwen/Qwen2.5-7B-Instruct --ctx 16384 --samples 3")
        ok = True   # nothing failed; just incomplete
    elif p2 is False:
        print("VERDICT: NOT SAFE / INCONCLUSIVE — Part 2 did not cleanly pass (OOM, incomplete,")
        print("  or recall regressed). Do NOT change the default. Fix the Part 2 failures above")
        print("  (e.g. lower --ctx/--samples so runs don't OOM) and re-run.")
        ok = False
    else:
        print("VERDICT: SAFE to make Gram-eigh the default.")
        print("  - Math equivalent to SVD (Part 1); compress faster + recall held (Part 2).")
        print("  TO MAKE DEFAULT: in lowrank.py:_compress_layer_blocks_gpu_inner change the gate")
        print("    `if os.environ.get('DIFFKV_COMPRESS_GRAM_SVD','0')=='1'` so Gram-eigh runs")
        print("    unless DIFFKV_COMPRESS_GRAM_SVD=0 is explicitly set (i.e. default to '1').")
        print("    The r_proj<=32 recipe is a fidelity trade — default it ONLY if gram_rproj32")
        print("    also showed SAFE above.")
        ok = True
    print("=" * 74)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
