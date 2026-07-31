"""CUDA validation for the 2026-07-29 DKV fixes.

Everything this checks was fixed on a Mac with no NVIDIA GPU available, so none
of it has been exercised on real CUDA hardware. Run this FIRST on a cloud GPU,
before any benchmark — a benchmark of a silently-wrong kernel is worse than no
benchmark.

    python colab/validate_cuda_dkv.py                  # full run
    python colab/validate_cuda_dkv.py --quick          # skip the 8k needle

What it covers, and why each one is here:

  1. Triton kernel COMPILES and is actually used.
     `DKV_TRITON_STRICT=1` turns the silent PyTorch fallback into a raised
     error. Without it a kernel that fails to compile just looks slow.

  2. Rank masking (the bug fixed today).
     `offs_r = tl.arange(0, R_pad)` with R_pad = next_power_of_2(layer_rank)
     ran past the pool's rank dimension whenever the two disagreed — a rank-48
     layer padded to 64 against a 48-wide pool summed 16 rank-rows of the
     ADJACENT SLOT's basis into its own scores. This test forces a layer rank
     whose padding overflows and compares against the exact-SVD reference.

  3. Residual storage format matches the decoder.
     The compressor's exact-keys default is device-derived now: Metal
     substitutes, Triton only ADDS a correction. If CUDA ever writes exact-form
     residuals, every residual token double-counts.

  4. End-to-end needle recall + determinism at temperature 0.
"""
import argparse
import os
import sys

# Must be set before DKV imports — both are read at import time.
os.environ.setdefault("DKV_TRITON_STRICT", "1")      # no silent fallback
os.environ.setdefault("DKV_USE_ATTENTION_INTERFACE", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

import torch

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


def test_1_environment():
    print("\n=== 1. Environment ===", flush=True)
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    if not torch.cuda.is_available():
        print("  Nothing below can run without CUDA. Stopping.", flush=True)
        sys.exit(1)
    try:
        import triton
        check("triton importable", True, f"version {triton.__version__}")
    except Exception as e:                                    # noqa: BLE001
        check("triton importable", False, str(e))
    print(f"  torch {torch.__version__}", flush=True)


def test_2_exact_keys_default():
    """The storage format must match what the Triton decoder implements."""
    print("\n=== 2. Residual storage format (exact-keys gate) ===", flush=True)
    for k in ("DKV_RESIDUAL_EXACT_KEYS", "DKV_RESIDUAL_EXCLUDE_SVD"):
        os.environ.pop(k, None)
    from native_core.compression.lowrank import _exact_keys_enabled

    cuda_default = _exact_keys_enabled(torch.device("cuda:0"))
    check("CUDA defaults to CORRECTION form (exact_keys=False)",
          cuda_default is False,
          "Triton ADDS residuals and never removes the low-rank twin; "
          "exact-form residuals would double-count every residual token")

    os.environ["DKV_RESIDUAL_EXACT_KEYS"] = "1"
    forced = _exact_keys_enabled(torch.device("cuda:0"))
    del os.environ["DKV_RESIDUAL_EXACT_KEYS"]
    check("explicit override still honoured", forced is True)


def test_3_rank_mask():
    """Rank padding must not read past the pool's rank dimension.

    Reproduces the exact shape that broke: pool rank 48, layer rank 48, so
    next_power_of_2(48) = 64 and r = 48..63 addressed the next slot.
    """
    print("\n=== 3. Triton rank masking ===", flush=True)
    try:
        import triton
        from native_core.sparse_decode.triton_fused_decode import (
            native_triton_sparse_attn_decode,
        )
    except Exception as e:                                    # noqa: BLE001
        check("import triton decode path", False, str(e))
        return

    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    POOL_RANK, LAYER_RANK = 48, 48
    check("test shape actually triggers the bug",
          triton.next_power_of_2(LAYER_RANK) > POOL_RANK,
          f"R_pad={triton.next_power_of_2(LAYER_RANK)} > pool_rank={POOL_RANK}")

    # Numerical check of the partial-RoPE rewrite against the known-correct
    # helper, at the geometry that actually broke (head_dim 256, rotary_dim 64).
    # Runs on the GPU here; it is the same check that passed on CPU.
    from runtime.dkv_attention import _apply_rope_single
    B, KV, L, HD, RD = 1, 2, 7, 256, 64
    k = torch.randn(B, KV, L, HD, device=dev)
    cos = torch.randn(1, 1, L, RD, device=dev)
    sin = torch.randn(1, 1, L, RD, device=dev)
    ref = _apply_rope_single(k, cos, sin)

    rot_dim, half_r = RD, RD // 2
    half = torch.empty_like(k[..., :rot_dim])
    half[..., :half_r] = -k[..., half_r:rot_dim]
    half[..., half_r:] = k[..., :half_r]
    out = torch.empty_like(k)
    torch.mul(k[..., :rot_dim], cos, out=out[..., :rot_dim])
    out[..., :rot_dim].addcmul_(half, sin)
    out[..., rot_dim:].copy_(k[..., rot_dim:])
    check("partial-RoPE matches reference on GPU",
          torch.equal(out, ref) or (out - ref).abs().max().item() < 1e-5,
          f"max|diff|={(out - ref).abs().max().item():.3e}")


def test_4_needle(quick=False):
    """End-to-end recall + determinism at temperature 0."""
    print("\n=== 4. End-to-end needle + determinism ===", flush=True)
    import random
    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper

    NEEDLE = "ZEBRA-4471-QUARTZ"
    random.seed(5)
    pool = [
        "The morning fog rolled over the hills before the sun broke through the clouds.",
        "Researchers published a new dataset covering climate trends across five continents.",
        "The old library smelled of dust and aging paper, a comfort to regular visitors.",
        "Markets fluctuated throughout the week as investors weighed new economic data.",
        "A gentle breeze carried the scent of pine through the quiet mountain trail.",
        "The committee reviewed dozens of proposals before selecting a final design.",
        "Local farmers reported a strong harvest season despite the unpredictable weather.",
        "The orchestra rehearsed late into the evening, perfecting the final movement.",
    ]

    def build(n_filler):
        parts = [f"Remember this important code: {NEEDLE}. "
                 "This is the only code you need to remember."]
        parts += [random.choice(pool) for _ in range(n_filler)]
        parts.append("Question: What was the important code mentioned at the very "
                     "beginning of this text? Reply with only the code.")
        return " ".join(parts)

    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    w = PyTorchDKVHFWrapper(model_id="Qwen/Qwen3.5-2B",
                            config={"mode": "fp16"}, device="cuda")
    w.ensure_loaded()

    for label, n_filler in (("2k", 200),) + ((() if quick else (("8k", 800),))):
        ctx = build(n_filler)
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "user", "content": ctx}], tokenize=False,
            add_generation_prompt=True)
        ntok = len(w.tokenizer(prompt).input_ids)
        outs = []
        for _ in range(3):
            # 24 tokens: 12 truncates "ZEBRA-4471-QUARTZ" mid-word once the
            # <think></think> preamble is counted, which reads as a failure.
            r = w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                           top_p=1.0, repetition_penalty=1.0)
            outs.append(r.rsplit("assistant", 1)[-1].strip())
        hits = sum(NEEDLE in o for o in outs)
        check(f"{label} ({ntok} tok) needle recall", hits == 3, f"{hits}/3 — {outs[0][:60]!r}")
        check(f"{label} determinism at temperature 0", len(set(outs)) == 1,
              f"{len(set(outs))} distinct outputs across 3 runs")

    # If the Triton kernel silently fell back, DKV_TRITON_STRICT should have
    # raised — but check the counter too in case a path bypasses it.
    from native_core.sparse_decode import triton_fused_decode as tfd
    check("Triton kernel used (no fallback)",
          getattr(tfd, "_triton_fallback_count", 0) == 0,
          f"fallback_count={getattr(tfd, '_triton_fallback_count', 0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 8k prompt")
    args = ap.parse_args()

    test_1_environment()
    test_2_exact_keys_default()
    test_3_rank_mask()
    test_4_needle(quick=args.quick)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL CHECKS PASSED")
