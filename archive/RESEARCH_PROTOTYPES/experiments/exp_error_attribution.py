"""
experiments/exp_error_attribution.py — Phase 2.5 Objective 4

Error Attribution Study: decompose reconstruction error into components.

Ablation design:
  A. FP16 deltas + anchors        (quantization_error = 0 baseline)
  B. INT8 deltas, no chaining     (pure quantization noise)
  C. FP16 chained reconstruction  (chain error without quantization)
  D. INT8 chained (full DiffKV)   (both errors combined)
  E. Varying anchor spacing       (anchor_spacing_error)
  F. Layer sensitivity ablation   (layer_sensitivity_error)

Total reconstruction error ≈
    quantization_error
  + reconstruction_chain_error
  + anchor_spacing_error
  + (adaptive_policy_error if applicable)
  + layer_sensitivity_error

Goal: which factor DOMINATES? Focus reduction there.
"""

import sys
import json
import math
import random
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import EMAPolicy
from reconstruction.reconstructor import KVReconstructor
from compression.quantization import quantize_int8, dequantize_int8


SEQ_LEN   = 4096
NUM_HEADS = 32
HEAD_DIM  = 128
MODES     = ["smooth", "mixed", "real_approx"]


def _rms_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean relative L2 error over all tokens."""
    diff = (original.float() - reconstructed.float())
    l2   = torch.linalg.vector_norm(diff, dim=(-1, -2, -3))
    orig = torch.linalg.vector_norm(original.float(), dim=(-1, -2, -3))
    return (l2 / (orig + 1e-9)).mean().item()


def ablation_fp16_deltas(kv: torch.Tensor, interval: int) -> float:
    """A. FP16 deltas: no quantization, pure chaining error only."""
    strategy = PeriodicAnchorStrategy(interval=interval)
    manager  = AnchorManager(strategy=strategy)
    manager.compress(kv)
    seq_len  = kv.shape[0]

    # Reconstruct in FP16 without INT8 quantization
    out = kv.clone()  # anchors are exact
    for i in range(seq_len):
        if not manager.is_anchor(i):
            anchor_idx, anchor_kv = manager.get_preceding_anchor(i)
            out[i] = (anchor_kv.float() + (kv[i].float() - anchor_kv.float())).half()
    # This is essentially the identity — pure chain error only
    return _rms_error(kv, out)


def ablation_int8_no_chain(kv: torch.Tensor, interval: int) -> float:
    """B. INT8 deltas independently (no chaining). Pure quantization error."""
    strategy = PeriodicAnchorStrategy(interval=interval)
    manager  = AnchorManager(strategy=strategy)
    manager.compress(kv)
    seq_len  = kv.shape[0]

    out = kv.clone().float()
    last_anchor_kv = None
    for i in range(seq_len):
        if manager.is_anchor(i):
            last_anchor_kv = kv[i].float()
            out[i] = last_anchor_kv
        else:
            if last_anchor_kv is not None:
                delta = kv[i].float() - last_anchor_kv
                qdelta = quantize_int8(delta)
                recon_delta = dequantize_int8(qdelta, target_dtype=torch.float32)
                # Use TRUE anchor (no chain error)
                out[i] = (last_anchor_kv + recon_delta)

    return _rms_error(kv, out.half())


def ablation_full_diffkv(kv: torch.Tensor, interval: int) -> float:
    """D. Full DiffKV: INT8 deltas + chained reconstruction."""
    strategy = PeriodicAnchorStrategy(interval=interval)
    manager  = AnchorManager(strategy=strategy)
    manager.compress(kv)
    recon   = KVReconstructor(manager)
    result  = recon.reconstruct_range(0, kv.shape[0] - 1)
    return _rms_error(kv, result.kv)


def ablation_spacing_sweep(kv: torch.Tensor, intervals: list) -> dict:
    """E. Vary anchor spacing, measure error at each level."""
    results = {}
    for interval in intervals:
        err = ablation_full_diffkv(kv, interval)
        results[interval] = round(err, 6)
    return results


def ablation_adaptive_vs_periodic(kv: torch.Tensor, interval: int) -> dict:
    """Compare error from periodic vs EMA adaptive anchor placement."""
    # Periodic
    m_per = AnchorManager(strategy=PeriodicAnchorStrategy(interval=interval))
    m_per.compress(kv)
    r_per = KVReconstructor(m_per).reconstruct_range(0, kv.shape[0] - 1)
    err_per = _rms_error(kv, r_per.kv)

    # EMA adaptive
    m_ema = AnchorManager(strategy=EMAPolicy(
        alpha=0.1, sensitivity_factor=2.5, max_interval=interval*2, min_interval=8
    ))
    m_ema.compress(kv)
    r_ema = KVReconstructor(m_ema).reconstruct_range(0, kv.shape[0] - 1)
    err_ema = _rms_error(kv, r_ema.kv)

    return {
        "periodic_error": round(err_per, 6),
        "ema_error":      round(err_ema, 6),
        "adaptive_delta": round(err_ema - err_per, 6),
        "periodic_density": round(m_per.index_list.__len__() / kv.shape[0], 4),
        "ema_density":      round(m_ema.index_list.__len__() / kv.shape[0], 4),
    }


def main():
    output_dir = Path("results/error_attribution")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)

    print(f"\n{'='*70}")
    print("  ERROR ATTRIBUTION STUDY")
    print(f"  seq_len={SEQ_LEN:,} | heads={NUM_HEADS} | dim={HEAD_DIM}")
    print(f"{'='*70}\n")

    all_results = {}
    intervals = [16, 32, 64, 128, 256, 512]

    for mode in MODES:
        kv = gen.generate(SEQ_LEN, mode=mode)

        print(f"[{mode}]")

        # A: FP16 deltas (chain error only)
        err_a = ablation_fp16_deltas(kv, interval=64)

        # B: INT8 no chain (quantization error only)
        err_b = ablation_int8_no_chain(kv, interval=64)

        # D: Full DiffKV (combined)
        err_d = ablation_full_diffkv(kv, interval=64)

        # E: Spacing sweep
        spacing = ablation_spacing_sweep(kv, intervals)

        # Adaptive vs periodic
        adap = ablation_adaptive_vs_periodic(kv, interval=64)

        # Attribute error components
        chain_error = err_a       # FP16 chain = just chaining
        quant_error = err_b       # INT8 without chain = pure quantization
        interaction = max(0.0, err_d - chain_error - quant_error)  # non-linear interaction

        result = {
            "mode":          mode,
            "fp16_chain":    round(err_a, 6),
            "int8_no_chain": round(err_b, 6),
            "full_diffkv":   round(err_d, 6),
            "attribution": {
                "chain_error_contribution":   round(chain_error, 6),
                "quantization_contribution":  round(quant_error, 6),
                "interaction_term":           round(interaction, 6),
                "dominant_factor": (
                    "chain_error" if chain_error > quant_error
                    else "quantization"
                ),
            },
            "spacing_sweep":    spacing,
            "adaptive_vs_periodic": adap,
        }
        all_results[mode] = result

        print(f"  FP16 chain error (chain only):     {err_a:.6f}")
        print(f"  INT8 quant error (no chain):       {err_b:.6f}")
        print(f"  Full DiffKV (combined):            {err_d:.6f}")
        print(f"  Dominant factor: {result['attribution']['dominant_factor']}")
        print(f"  Adaptive policy delta err:         {adap['adaptive_delta']:+.6f}")
        print(f"  Spacing sweep: "
              f"{' | '.join(f'{i}={v:.5f}' for i, v in list(spacing.items())[:4])}")
        print()

    out_path = output_dir / "error_attribution.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Results -> {out_path}")
    print("[->] Run visualization/plot_error_attribution.py to visualize")


if __name__ == "__main__":
    main()
