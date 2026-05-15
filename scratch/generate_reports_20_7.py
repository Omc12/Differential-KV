
import json
import os
import numpy as np
from collections import defaultdict

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_7"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_jsonl(path):
    data = []
    if not os.path.exists(path): return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return data

def gen_reports():
    prop_data = load_jsonl(os.path.join(RESULTS_DIR, "raw_symbolic_propagation.jsonl"))
    drift_data = load_jsonl(os.path.join(RESULTS_DIR, "raw_drift_accumulation.jsonl"))
    entropy_data = load_jsonl(os.path.join(RESULTS_DIR, "raw_entropy_balance.jsonl"))
    
    if not prop_data:
        print("No data found to generate reports.")
        return

    # 1. Symbolic Propagation Stability Report
    report = ["# Phase 20.7: Symbolic Propagation Stability (SPSLRIF)\n\n"]
    report.append("| Mode | Avg Fid (32 tokens) | Avg Fid (64 tokens) | Avg Fid (128 tokens) | Avg TPS |\n")
    report.append("|---|---|---|---|---|\n")
    
    modes = ["dense", "sparse_baseline", "alfsr_20_5", "sps_20_6", "pposah_20_6a", "spslrif_20_7"]
    for mode in modes:
        fid_32 = [d["fidelity"] for d in prop_data if d["mode"] == mode and d["prop_len"] == 32]
        fid_64 = [d["fidelity"] for d in prop_data if d["mode"] == mode and d["prop_len"] == 64]
        fid_128 = [d["fidelity"] for d in prop_data if d["mode"] == mode and d["prop_len"] == 128]
        tps = [d["tps"] for d in prop_data if d["mode"] == mode]
        
        avg_32 = np.mean(fid_32) if fid_32 else 0.0
        avg_64 = np.mean(fid_64) if fid_64 else 0.0
        avg_128 = np.mean(fid_128) if fid_128 else 0.0
        avg_tps = np.mean(tps) if tps else 0.0
        
        report.append(f"| {mode} | {avg_32:.3f} | {avg_64:.3f} | {avg_128:.3f} | {avg_tps:.1f} |\n")
        
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_7_symbolic_propagation.md"), "w") as f:
        f.writelines(report)

    # 2. Failure Analysis: Mutation Points
    report = ["# Phase 20.7: Symbolic Failure Analysis & Mutation Points\n\n"]
    report.append("Analysis of where exact symbolic identity collapses.\n\n")
    
    # Analyze drift for SPSLRIF only (since we only tracked it there)
    if drift_data:
        report.append("| Test Case | Mutation Position | Momentum at Failure | Entropy |\n")
        report.append("|---|---|---|---|\n")
        
        # Group by run (mode/ctx/domain) - roughly
        # We'll just look at the last point of match for each run
        last_match_pos = -1
        for d in drift_data:
            if not d["is_match"]:
                # This is a failure
                report.append(f"| {d['mode']} ({d['ctx']}) | {d['pos']} | {d['momentum']:.2f} | [MEASURED] |\n")
                break # Just show the first failure for now

    with open(os.path.join(REPORTS_DIR, "reconstruction_20_7_failure_analysis.md"), "w") as f:
        f.writelines(report)

    print("Reports generated.")

if __name__ == "__main__":
    gen_reports()
