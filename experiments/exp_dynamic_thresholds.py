"""
experiments/exp_dynamic_thresholds.py — Task 3: Dynamic Threshold Evolution

Specifically tests how thresholds evolve over time during sequence traversal.
Tracks:
  - Delta RMS (the signal)
  - Evolving Threshold (the adaptive gate)
  - Anchor trigger points

Compares:
  1. Static Threshold (Phase 1 baseline)
  2. Rolling Window Threshold
  3. Adaptive Percentile Threshold (Targeting 5% rate)
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kv_generator import KVGenerator
from anchor_logic.anchor_manager import AnchorManager
from anchor_logic.adaptive_policies import DynamicThresholdPolicy, AbsoluteNormalizedPolicy


SEQ_LEN   = 2048   # Long enough to see evolution
NUM_HEADS = 32
HEAD_DIM  = 128
MODE      = "mixed"


def run_dynamic_trace(policy, kv):
    """Run compression and return the token-by-token trace of RMS and Threshold."""
    manager = AnchorManager(strategy=policy)
    manager.compress(kv)

    # We need to reach into the policy to get history
    # This requires the policy to have tracked it (which our Phase 2 policies do)
    rms_history = []
    threshold_history = []
    anchor_indices = manager.index_list

    # Re-run manually or extract from policy if it stored them
    # Our policies store histories in their get_stats() or internal lists
    stats = policy.get_stats()

    # For DynamicThresholdPolicy, we can get history from the tracker
    if hasattr(policy, "tracker") and hasattr(policy.tracker, "get_history"):
        threshold_history = policy.tracker.get_history()
    else:
        # For AbsoluteNormalized, it's a constant
        thresh = getattr(policy, "threshold", 0.3)
        threshold_history = [thresh] * SEQ_LEN

    # Re-calculate RMS for the plot
    last_anchor_kv = kv[0].float()
    for i in range(SEQ_LEN):
        if i in manager.anchors:
            last_anchor_kv = kv[i].float()
            rms_history.append(0.0) # Anchor point
        else:
            delta = kv[i].float() - last_anchor_kv
            rms = (delta.norm() / np.sqrt(delta.numel())).item()
            rms_history.append(rms)

    return {
        "rms": rms_history,
        "threshold": threshold_history,
        "anchors": anchor_indices,
        "stats": stats
    }


def main():
    output_dir = Path("results/dynamic_thresholds")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = KVGenerator(num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=42)
    kv  = gen.generate(SEQ_LEN, mode=MODE)

    print(f"\n{'='*70}")
    print("  DYNAMIC THRESHOLD EVOLUTION EXPERIMENT")
    print(f"{'='*70}\n")

    policies = {
        "Static-0.3":      AbsoluteNormalizedPolicy(threshold=0.3),
        "Rolling-k2":      DynamicThresholdPolicy(tracker_type="rolling", k=2.0, window_size=64),
        "Adaptive-5pct":   DynamicThresholdPolicy(tracker_type="adaptive", target_rate=0.05, buffer_size=128),
    }

    traces = {}
    for label, policy in policies.items():
        print(f"  Running {label}...")
        traces[label] = run_dynamic_trace(policy, kv)

    # Save
    out_path = output_dir / "threshold_traces.json"
    # Truncate lists for JSON if they are huge, but 2048 is fine
    with open(out_path, "w") as f:
        json.dump(traces, f, indent=2)

    print(f"\n[OK] Traces saved -> {out_path}")
    print(f"[->] Run visualization/plot_dynamic_thresholds.py to see the evolution")


if __name__ == "__main__":
    main()
