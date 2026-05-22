"""
experiments/exp_real_kv.py — Task 5: Real Model KV Collection

Captures real KV tensors from a small HuggingFace model (GPT2 default)
and compares their statistical properties to synthetic KV approximations.

Usage:
    python experiments/exp_real_kv.py                       # uses GPT2
    python experiments/exp_real_kv.py --model tinyllama
    python experiments/exp_real_kv.py --model gpt2 --no-compress
"""

import sys
import json
import argparse
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kv_collection.hf_collector import HFKVCollector, PROMPT_LIBRARY
from kv_collection.kv_dataset import KVDataset
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.strategies import PeriodicAnchorStrategy
from anchor_logic.adaptive_policies import EMAPolicy, RollingVariancePolicy
from reconstruction.reconstructor import KVReconstructor
from benchmarks.kv_generator import KVGenerator


def compress_and_report(kv_by_layer, strategy, label):
    """Compress all layers with strategy, report average stats."""
    ratios, errors, densities = [], [], []
    import random; random.seed(0)

    for layer_idx, kv in kv_by_layer.items():
        seq_len = kv.shape[0]
        if seq_len < 10:
            continue

        manager = AnchorManager(strategy=strategy)
        stats   = manager.compress(kv)
        recon   = KVReconstructor(manager)

        window = min(32, seq_len // 4)
        layer_errors = []
        for _ in range(min(10, seq_len // window)):
            s = random.randint(0, max(0, seq_len - window - 1))
            e = min(s + window - 1, seq_len - 1)
            err = recon.measure_error(kv, s, e)
            layer_errors.append(err["mean_relative"])

        ratios.append(stats.compression_ratio)
        densities.append(stats.anchor_density)
        if layer_errors:
            errors.append(sum(layer_errors) / len(layer_errors))

    if not ratios:
        return {}

    return {
        "strategy":          label,
        "mean_ratio":        round(sum(ratios) / len(ratios), 4),
        "mean_density":      round(sum(densities) / len(densities), 4),
        "mean_error":        round(sum(errors) / len(errors), 6) if errors else 0.0,
        "min_ratio":         round(min(ratios), 4),
        "max_ratio":         round(max(ratios), 4),
        "layers_analyzed":   len(ratios),
    }


def analyze_kv_statistics(kv_by_layer):
    """Compute basic statistics on real KV tensors."""
    all_means, all_stds, all_rms = [], [], []
    smoothness_scores = []

    for layer_idx, kv in kv_by_layer.items():
        f = kv.float()
        all_means.append(f.mean().item())
        all_stds.append(f.std().item())
        rms = (f.norm() / math.sqrt(f.numel())).item()
        all_rms.append(rms)

        # Consecutive-token smoothness
        if kv.shape[0] > 1:
            consec = []
            for i in range(1, min(kv.shape[0], 100)):
                d = kv[i].float() - kv[i-1].float()
                consec.append((d.norm() / math.sqrt(d.numel())).item())
            smoothness_scores.append(sum(consec) / len(consec))

    n = len(all_means)
    return {
        "num_layers":        n,
        "mean_activation":   round(sum(all_means) / n, 5) if n else 0,
        "mean_std":          round(sum(all_stds) / n, 5)  if n else 0,
        "mean_rms":          round(sum(all_rms) / n, 5)   if n else 0,
        "mean_smoothness":   round(sum(smoothness_scores) / len(smoothness_scores), 5)
                             if smoothness_scores else 0,
    }


def compare_with_synthetic(real_kv_by_layer, num_heads, head_dim, seq_len):
    """Compare real KV stats to each synthetic mode."""
    gen = KVGenerator(num_heads=num_heads, head_dim=head_dim, seed=42)
    real_stats = analyze_kv_statistics(real_kv_by_layer)

    print("\n  Real KV statistics (averaged across layers):")
    for k, v in real_stats.items():
        print(f"    {k:<22}: {v}")

    print("\n  Synthetic KV statistics (for comparison):")
    for mode in ["gaussian", "smooth", "mixed", "real_approx"]:
        try:
            syn_kv  = gen.generate(min(seq_len, 512), mode=mode)
            syn_kv_dict = {0: syn_kv}
            syn_stats = analyze_kv_statistics(syn_kv_dict)
            print(f"    [{mode:<12}] mean_rms={syn_stats['mean_rms']:.4f}  "
                  f"smoothness={syn_stats['mean_smoothness']:.4f}")
        except Exception as e:
            print(f"    [{mode}] error: {e}")

    return real_stats


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  REAL MODEL KV COLLECTION — {args.model.upper()}")
    print(f"{'='*65}\n")

    # Build prompt list from library
    prompts, text_types = [], []
    for ttype, plist in PROMPT_LIBRARY.items():
        for p in plist[:args.prompts_per_type]:
            prompts.append(p)
            text_types.append(ttype)

    print(f"  Prompts to run: {len(prompts)} ({list(PROMPT_LIBRARY.keys())})")

    # Load model and collect
    collector = HFKVCollector(
        model_name=args.model,
        device="auto",
        max_layers=args.max_layers,
        fp16=True,
    )

    try:
        collector.load_model()
    except Exception as e:
        print(f"\n[ERROR] Could not load model '{args.model}': {e}")
        print("  Make sure transformers is installed: pip install transformers")
        print("  Then retry or choose a smaller model: gpt2, opt-125m")
        return

    snapshots = collector.collect(prompts, text_types=text_types)

    if not snapshots:
        print("[ERROR] No KV snapshots captured. Check model compatibility.")
        return

    # Save raw snapshots
    snap_dir = output_dir / args.model
    collector.save(snapshots, str(snap_dir))

    # Load and analyze
    dataset = KVDataset(str(snap_dir))
    print(f"\n  Loaded: {dataset}")

    all_results = []

    strategies = {
        "Periodic-64":  PeriodicAnchorStrategy(interval=64),
        "EMA-balanced": EMAPolicy(alpha=0.1, sensitivity_factor=2.5,
                                   max_interval=256, min_interval=8),
        "Rolling-k3":   RollingVariancePolicy(k=3.0, window_size=64,
                                               max_interval=256, min_interval=8),
    }

    for meta, kv_by_layer in dataset.iter_snapshots():
        print(f"\n  [{meta['text_type']}] seq={meta['seq_len']} "
              f"layers={meta['num_layers']}")

        # Statistics
        kv_stats = analyze_kv_statistics(kv_by_layer)
        result   = {"meta": meta, "kv_stats": kv_stats, "compression": {}}

        if not args.no_compress:
            for label, strategy in strategies.items():
                comp = compress_and_report(kv_by_layer, strategy, label)
                result["compression"][label] = comp
                print(f"    {label:<16}: ratio={comp.get('mean_ratio', 0):.3f}  "
                      f"density={comp.get('mean_density', 0):.4f}  "
                      f"err={comp.get('mean_error', 0):.5f}")

        all_results.append(result)

    # Compare with synthetic
    if snapshots:
        first_snap = snapshots[0]
        compare_with_synthetic(
            first_snap.kv_by_layer,
            first_snap.num_heads,
            first_snap.head_dim,
            first_snap.seq_len,
        )

    # Save analysis
    out_path = output_dir / f"{args.model}_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Analysis saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real KV Collection Experiment")
    parser.add_argument("--model",            default="gpt2",
                        choices=["gpt2", "gpt2-med", "opt-125m", "tinyllama", "phi2"])
    parser.add_argument("--max-layers",       type=int, default=None)
    parser.add_argument("--prompts-per-type", type=int, default=1)
    parser.add_argument("--no-compress",      action="store_true")
    parser.add_argument("--output",           default="results/real_kv/")
    args = parser.parse_args()
    main(args)
