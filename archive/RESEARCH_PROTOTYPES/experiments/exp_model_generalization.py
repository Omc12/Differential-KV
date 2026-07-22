"""
experiments/exp_model_generalization.py — Phase 2.5 Objective 5

Real-model generalization study.

Tests whether Differential KV statistical assumptions generalize
across multiple transformer architectures.

Models tested (in order of increasing size):
  - GPT2
  - opt-125m (if GPT2 fails)
  - TinyLlama (if VRAM permits)

Prompt domains:
  - prose
  - code
  - reasoning
  - repetitive
  - multilingual (basic)

Measures per model × domain:
  - smoothness (consecutive token RMS)
  - kurtosis
  - sparsity at 0.1
  - lag-1 autocorrelation
  - rank for 90% SVD energy
  - anchor density (EMA-balanced)
  - compression ratio (EMA-balanced)
  - reconstruction error

Goal: determine universality of DKV statistical assumptions
"""

import sys
import json
import argparse
import math
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_collection.hf_collector import HFKVCollector, PROMPT_LIBRARY
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import EMAPolicy
from reconstruction.reconstructor import KVReconstructor
from analysis.lowrank_analyzer import LowRankAnalyzer

MULTILINGUAL_PROMPTS = [
    "La inteligencia artificial es un campo de la informatica que busca crear maquinas capaces de realizar tareas que normalmente requieren inteligencia humana.",
    "L'intelligence artificielle est un domaine de l'informatique qui vise a creer des machines capables d'effectuer des taches qui necessitent normalement l'intelligence humaine.",
]


def analyze_kv_distribution(kv: torch.Tensor) -> dict:
    """Comprehensive statistical analysis of one KV tensor."""
    f = kv.float()
    flat = f.flatten()

    # Basic stats
    mean = float(flat.mean())
    std  = float(flat.std())
    rms  = float((f.norm() / math.sqrt(f.numel())).item())

    # Sparsity
    sparsity_01 = float((flat.abs() < 0.1).float().mean())

    # Kurtosis
    if std > 1e-9:
        kurtosis = float(((((flat - flat.mean()) / std) ** 4).mean() - 3))
    else:
        kurtosis = 0.0

    # Smoothness (consecutive token RMS)
    consec_rms = []
    for i in range(1, min(kv.shape[0], 200)):
        diff = kv[i].float() - kv[i-1].float()
        consec_rms.append((diff.norm() / math.sqrt(diff.numel())).item())
    smoothness = float(np.mean(consec_rms)) if consec_rms else 0.0

    # Autocorrelation
    if len(consec_rms) > 2:
        arr = np.array(consec_rms)
        if arr.std() > 1e-9:
            autocorr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    return {
        "mean":        round(mean,        5),
        "std":         round(std,         5),
        "rms":         round(rms,         5),
        "sparsity_01": round(sparsity_01, 4),
        "kurtosis":    round(kurtosis,    4),
        "smoothness":  round(smoothness,  5),
        "autocorr":    round(autocorr,    4),
    }


def compress_and_measure(kv: torch.Tensor, strategy) -> dict:
    """Compress a KV tensor and return key metrics."""
    import random; random.seed(0)
    manager = AnchorManager(strategy=strategy)
    stats   = manager.compress(kv)
    recon   = KVReconstructor(manager)
    seq_len = kv.shape[0]
    window  = min(32, seq_len // 4)
    errors  = []
    for _ in range(min(10, seq_len // window)):
        s = random.randint(0, max(0, seq_len - window - 1))
        e = min(s + window - 1, seq_len - 1)
        err = recon.measure_error(kv, s, e)
        errors.append(err["mean_relative"])
    return {
        "compression_ratio": round(stats.compression_ratio, 4),
        "anchor_density":    round(stats.anchor_density, 4),
        "mean_error":        round(float(np.mean(errors)), 6) if errors else 0.0,
    }


def analyze_snapshot(model_name: str, text_type: str,
                     kv_by_layer: dict) -> dict:
    """Full analysis of one KV snapshot."""
    lr_analyzer = LowRankAnalyzer()
    strategy_periodic = PeriodicAnchorStrategy(interval=64)
    strategy_ema      = EMAPolicy(alpha=0.1, sensitivity_factor=2.5,
                                   max_interval=256, min_interval=8)

    per_layer_stats = []
    for layer_idx, kv in list(kv_by_layer.items())[:8]:  # first 8 layers
        dist = analyze_kv_distribution(kv)

        # Compression
        comp_periodic = compress_and_measure(kv, PeriodicAnchorStrategy(interval=64))
        comp_ema      = compress_and_measure(kv, EMAPolicy(
            alpha=0.1, sensitivity_factor=2.5, max_interval=256, min_interval=8
        ))

        # Low-rank
        mgr = AnchorManager(strategy=strategy_periodic)
        mgr.compress(kv)
        lr_profile = lr_analyzer.analyze(kv, mgr.index_list, window_size=64)
        lr_summary = lr_profile.summary()

        per_layer_stats.append({
            "layer": layer_idx,
            "dist":  dist,
            "periodic": comp_periodic,
            "ema":      comp_ema,
            "rank_90pct": lr_summary["rank_for_90pct"],
            "drift":      lr_summary["mean_subspace_drift"],
        })

    # Aggregate across layers
    def _mean(key, sub=None):
        vals = [
            (s[sub][key] if sub else s[key])
            for s in per_layer_stats
            if (sub and key in s.get(sub, {})) or (not sub and key in s)
        ]
        return round(float(np.mean(vals)), 5) if vals else 0.0

    return {
        "model":          model_name,
        "text_type":      text_type,
        "num_layers":     len(kv_by_layer),
        "mean_rms":       _mean("rms", "dist"),
        "mean_kurtosis":  _mean("kurtosis", "dist"),
        "mean_sparsity":  _mean("sparsity_01", "dist"),
        "mean_smoothness": _mean("smoothness", "dist"),
        "mean_autocorr":  _mean("autocorr", "dist"),
        "mean_rank_90pct": _mean("rank_90pct"),
        "mean_subspace_drift": _mean("drift"),
        "periodic_ratio": _mean("compression_ratio", "periodic"),
        "ema_ratio":      _mean("compression_ratio", "ema"),
        "ema_error":      _mean("mean_error", "ema"),
        "ema_density":    _mean("anchor_density", "ema"),
        "per_layer":      per_layer_stats,
    }


def main(args):
    output_dir = Path("results/model_generalization")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  REAL-MODEL GENERALIZATION STUDY")
    print(f"  models={args.models}")
    print(f"{'='*70}\n")

    # Build prompts including multilingual
    prompts, text_types = [], []
    for ttype, plist in PROMPT_LIBRARY.items():
        for p in plist[:1]:
            prompts.append(p)
            text_types.append(ttype)
    for p in MULTILINGUAL_PROMPTS[:1]:
        prompts.append(p)
        text_types.append("multilingual")

    all_results = {}

    for model_name in args.models:
        print(f"\n[Model: {model_name}]")

        try:
            collector = HFKVCollector(model_name=model_name, device="auto",
                                      max_layers=8, fp16=True)
            collector.load_model()
            snapshots = collector.collect(prompts, text_types=text_types)
        except Exception as e:
            print(f"  [SKIP] Could not load {model_name}: {e}")
            continue

        model_results = []
        for snap in snapshots:
            print(f"  Analyzing: type={snap.text_type} seq={snap.seq_len} "
                  f"layers={snap.num_layers}")
            analysis = analyze_snapshot(model_name, snap.text_type, snap.kv_by_layer)
            model_results.append(analysis)
            print(f"    rms={analysis['mean_rms']:.4f}  "
                  f"kurt={analysis['mean_kurtosis']:.2f}  "
                  f"rank90={analysis['mean_rank_90pct']}  "
                  f"drift={analysis['mean_subspace_drift']:.4f}  "
                  f"ema_ratio={analysis['ema_ratio']:.3f}x  "
                  f"ema_err={analysis['ema_error']:.5f}")

        all_results[model_name] = model_results

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    out_path = output_dir / "model_generalization.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results -> {out_path}")

    # Cross-model comparison
    print(f"\n{'='*70}")
    print("  CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Model':<14} {'Type':<12} {'Kurtosis':>9} {'Rank90':>7} {'Drift':>8} {'EMA-Ratio':>10} {'EMA-Err':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model_name, model_results in all_results.items():
        for r in model_results:
            print(f"  {model_name:<14} {r['text_type']:<12} "
                  f"{r['mean_kurtosis']:>9.3f} "
                  f"{r['mean_rank_90pct']:>7} "
                  f"{r['mean_subspace_drift']:>8.4f} "
                  f"{r['ema_ratio']:>10.4f} "
                  f"{r['ema_error']:>9.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["gpt2"],
                        choices=["gpt2", "gpt2-med", "opt-125m", "tinyllama"])
    args = parser.parse_args()
    main(args)
