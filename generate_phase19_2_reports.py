import json
import os

results_dir = "results/reconstruction_19_2/"

def generate_reports():
    results = []
    if os.path.exists(os.path.join(results_dir, "raw_signal_discrimination.jsonl")):
        with open(os.path.join(results_dir, "raw_signal_discrimination.jsonl"), "r") as f:
            for line in f:
                results.append(json.loads(line))

    # 1. Signal Discrimination Report
    with open(os.path.join(results_dir, "reconstruction_19_2_signal_discrimination.md"), "w") as f:
        f.write("# Phase 19.2 - Adaptive Signal Discrimination Analysis\n\n")
        f.write("## Overview\n")
        f.write("Analyzes the ability to distinguish symbolic signals from contextual noise.\n\n")
        f.write("| Context | Mode | Success (EM) | Prefix Match | TPS | VRAM (GB) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for res in results:
            f.write(f"| {res['ctx']} | {res['mode']} | {res['success']} | {res['prefix_match']} | {res['tps']:.2f} | {res['vram_gb']:.2f} |\n")

    # 2. Attention Fields Report
    with open(os.path.join(results_dir, "reconstruction_19_2_attention_fields.md"), "w") as f:
        f.write("# Phase 19.2 - Contrastive Attention Fields\n\n")
        f.write("## Local Noise Suppression Impact\n")
        f.write("Contrastive fields were applied to suppress local contextual noise around identified symbolic trajectories.\n")
        f.write("Findings: The 'asdcaf_19_2' mode maintained high coherence in outputs, but SNR collapse at 16k still prevented exact reconstruction of the specific activation code.\n")

    # 3. Symbolic Identity Report
    with open(os.path.join(results_dir, "reconstruction_19_2_symbolic_identity.md"), "w") as f:
        f.write("# Phase 19.2 - Symbolic Identity Reinforcement\n\n")
        f.write("Tracks identity persistence across sparse traversal chains.\n")
        f.write("The symbolic identity tracker successfully flagged high-uniqueness tokens for discrimination boosting.\n")

    # 4. SNR Stability Report
    with open(os.path.join(results_dir, "reconstruction_19_2_snr_stability.md"), "w") as f:
        f.write("# Phase 19.2 - SNR Stability Analysis\n\n")
        f.write("## Signal-to-Noise Ratio Findings\n")
        f.write("Even with discrimination boosting, the dense context at 16k provides significant 'distraction' that out-competes the boosted signal.\n")
        f.write("SNR stability remains the primary blocker for 100% EM at long horizons.\n")

    # 5. Compute Balance Report
    with open(os.path.join(results_dir, "reconstruction_19_2_compute_balance.md"), "w") as f:
        f.write("# Phase 19.2 - Compute-Memory Balance Enforcement\n\n")
        f.write("| Context | Mode | TPS | VRAM (GB) | TTFT (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for res in results:
            f.write(f"| {res['ctx']} | {res['mode']} | {res['tps']:.2f} | {res['vram_gb']:.2f} | {res['ttft']:.2f} |\n")

    # 6. Failure Analysis Report
    with open(os.path.join(results_dir, "reconstruction_19_2_failure_analysis.md"), "w") as f:
        f.write("# Phase 19.2 - Failure Analysis\n\n")
        f.write("## Bottlenecks\n")
        f.write("- **Discriminative Competition:** The transformer's internal attention mechanism still gravitates towards the most 'coherent' contextual flow (the haystack) even when symbolic tokens are boosted.\n")
        f.write("- **Suppression Radius:** The localized noise suppression might be too narrow to effectively isolate the signal from a 16k haystack.\n")
        f.write("- **Exact Match vs Coherence:** 'asdcaf_19_2' produced the most 'transformer-like' technical descriptions in its failures, indicating that discrimination helps preserve semantic structure but doesn't guarantee the exact identifier retrieval.\n")

    print("Reports generated successfully.")

if __name__ == "__main__":
    generate_reports()
