
import json
import os
import numpy as np
import pandas as pd

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_8"
REPORTS_DIR = RESULTS_DIR # Requirements say they should be in the root of the results dir

def load_jsonl(name):
    path = os.path.join(RESULTS_DIR, name)
    data = []
    if not os.path.exists(path): return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return data

def gen_reports():
    print("Loading data for reports...")
    prop_data = load_jsonl("raw_symbolic_propagation.jsonl")
    density_data = load_jsonl("raw_attention_density.jsonl")
    integrity_data = load_jsonl("raw_delimiter_integrity.jsonl")
    focus_data = load_jsonl("raw_symbolic_focus.jsonl")
    hub_data = load_jsonl("raw_hub_registry.jsonl")
    random_data = load_jsonl("raw_random_propagation.jsonl")
    entropy_data = load_jsonl("raw_entropy_balance.jsonl")
    replay_data = load_jsonl("raw_replay_risk.jsonl")
    
    if not prop_data:
        print("No propagation data found. Skipping report generation.")
        return

    df = pd.DataFrame(prop_data)
    
    # 1. Attention Density Report
    if density_data:
        ddf = pd.DataFrame(density_data)
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_attention_density.md"), "w") as f:
            f.write("# Phase 20.8: Attention Density & Mass Dilution\n\n")
            f.write("## Metrics Summary\n")
            f.write(ddf.groupby("mode")[["density", "fragmentation", "peakiness"]].mean().to_markdown() + "\n\n")
            f.write("## Fragmentation by Context Length\n")
            f.write(ddf.pivot_table(index="mode", columns="ctx", values="fragmentation", aggfunc="mean").to_markdown() + "\n")

    # 2. Anchor Boosting Report
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_anchor_boosting.md"), "w") as f:
        f.write("# Phase 20.8: Structural Anchor Boosting Performance\n\n")
        f.write("## Fidelity Improvement (SPSLRIF vs SABEAF)\n")
        comp = df[df["mode"].isin(["spslrif_20_7", "sabeaf_20_8"])]
        if not comp.empty:
            f.write(comp.pivot_table(index="ctx", columns="mode", values="fidelity", aggfunc="mean").to_markdown() + "\n")

    # 3. Delimiter Integrity Report
    if integrity_data:
        idf = pd.DataFrame(integrity_data)
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_delimiter_integrity.md"), "w") as f:
            f.write("# Phase 20.8: Delimiter Integrity Field Analysis\n\n")
            f.write("## Drift Detection Frequency\n")
            f.write(idf.groupby("mode")["drift"].mean().to_markdown() + "\n")

    # 4. Hub Registry Report
    if hub_data:
        hdf = pd.DataFrame(hub_data)
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_hub_registry.md"), "w") as f:
            f.write("# Phase 20.8: HubAnchorRegistry Utilization\n\n")
            f.write("## Hub Registration Counts\n")
            f.write(hdf.groupby("mode")["registered_hubs"].max().to_markdown() + "\n")

    # 5. Random Propagation Report
    if random_data:
        rdf = pd.DataFrame(random_data)
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_random_propagation.md"), "w") as f:
            f.write("# Phase 20.8: Arbitrary Random Propagation\n\n")
            f.write("## Fidelity across Contexts\n")
            f.write(rdf.pivot_table(index="mode", columns="ctx", values="fidelity", aggfunc="mean").to_markdown() + "\n")

    # 6. Failure Analysis
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_failure_analysis.md"), "w") as f:
        f.write("# Phase 20.8: Failure Mode Decomposition\n\n")
        failures = df[df["exact_match"] == False]
        if not failures.empty:
            f.write("## Failure Distribution by Domain\n")
            f.write(failures.groupby("domain")["mode"].count().to_markdown() + "\n")

    # 7. Compute Balance Report
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_8_compute_balance.md"), "w") as f:
        f.write("# Phase 20.8: Compute & Throughput Balance\n\n")
        f.write("## TPS Analysis\n")
        f.write(df.pivot_table(index="mode", columns="ctx", values="tps", aggfunc="mean").to_markdown() + "\n")

    print("Scientific reports generated in results/reconstruction_20_8/")

if __name__ == "__main__":
    gen_reports()
