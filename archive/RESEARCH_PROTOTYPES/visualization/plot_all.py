"""
visualization/plot_all.py

One-shot script: runs ALL visualization scripts in sequence.
Assumes experiment results already exist in results/.

Usage:
    python visualization/plot_all.py
    python visualization/plot_all.py --results-dir results/
"""

import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PLOTS = [
    {
        "script": "visualization/plot_crossover.py",
        "input_key": "--input",
        "input_default": "results/crossover/crossover_mixed.json",
        "label": "Crossover Analysis",
    },
    {
        "script": "visualization/plot_compression.py",
        "input_key": "--input",
        "input_default": "results/compression/compression_results.json",
        "label": "Compression Analysis",
    },
    {
        "script": "visualization/plot_anchor_density.py",
        "input_key": "--input",
        "input_default": "results/anchor_density/periodic_sweep.json",
        "label": "Anchor Density Sweep",
    },
]


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Differential KV — Generate All Plots")
    print(f"{'='*60}\n")

    failed = []
    for plot in PLOTS:
        script = plot["script"]
        input_path = Path(args.results_dir) / Path(plot["input_default"]).name
        # Fall back to the default if the resolved path doesn't exist
        if not input_path.exists():
            input_path = Path(plot["input_default"])

        if not input_path.exists():
            print(f"  [SKIP] {plot['label']} — input not found: {input_path}")
            failed.append(plot["label"])
            continue

        print(f"  [→] {plot['label']} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, script,
             plot["input_key"], str(input_path),
             "--output", str(output_dir)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("done")
        else:
            print(f"FAILED\n{result.stderr[:500]}")
            failed.append(plot["label"])

    print(f"\n[OK] Plots saved → {output_dir}/")
    if failed:
        print(f"[!] Skipped (missing data): {', '.join(failed)}")
        print("    Run the experiments first:\n"
              "      python experiments/exp_crossover.py\n"
              "      python experiments/exp_compression.py\n"
              "      python experiments/exp_anchor_density.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output",      default="results/plots/")
    args = parser.parse_args()
    main(args)
