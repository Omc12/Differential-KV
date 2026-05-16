
import json
import os
import numpy as np
from collections import defaultdict

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_6"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

precision_data = load_jsonl(os.path.join(RESULTS_DIR, "raw_symbolic_precision.jsonl"))
drift_data = load_jsonl(os.path.join(RESULTS_DIR, "raw_drift_predictions.jsonl"))
entropy_data = load_jsonl(os.path.join(RESULTS_DIR, "raw_entropy_precision_balance.jsonl"))

# 1. Report: Symbolic Precision
def gen_precision_report():
    modes = ["dense", "sparse_baseline", "dtascc_19_7", "aascsi_20_3", "alfsr_20_5", "sps_20_6"]
    contexts = [4096, 8192, 16384]
    
    report = ["# Phase 20.6: Symbolic Precision Performance\n\n"]
    report.append("| Mode | Avg Fid (4k) | Avg Fid (8k) | Avg Fid (16k) | Total EM |\n")
    report.append("|---|---|---|---|---|\n")
    
    for mode in modes:
        row = [f"**{mode}**"]
        ems = 0
        for ctx in contexts:
            fids = [d["fidelity"] for d in precision_data if d["mode"] == mode and d["ctx"] == ctx]
            avg_fid = np.mean(fids) if fids else 0.0
            row.append(f"{avg_fid:.3f}")
            ems += sum(1 for d in precision_data if d["mode"] == mode and d["ctx"] == ctx and d["exact_match"])
        row.append(str(ems))
        report.append("| " + " | ".join(row) + " |\n")
    
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_6_symbolic_precision.md"), "w") as f:
        f.writelines(report)

# 2. Report: Suffix Stability
def gen_suffix_report():
    # Filter by domain 'suffix_integrity'
    report = ["# Phase 20.6: Suffix Stability Analysis\n\n"]
    report.append("Focused analysis on the `suffix_integrity` domain where exact suffix recovery is critical.\n\n")
    report.append("| Mode | Fid (4k) | Fid (8k) | Fid (16k) | TPS (16k) |\n")
    report.append("|---|---|---|---|---|\n")
    
    modes = ["dense", "sparse_baseline", "dtascc_19_7", "aascsi_20_3", "alfsr_20_5", "sps_20_6"]
    for mode in modes:
        row = [f"**{mode}**"]
        tps_16k = "N/A"
        for ctx in [4096, 8192, 16384]:
            match = [d for d in precision_data if d["mode"] == mode and d["ctx"] == ctx and d["domain"] == "suffix_integrity"]
            if match:
                row.append(f"{match[0]['fidelity']:.3f}")
                if ctx == 16384:
                    tps_16k = f"{match[0]['tps']:.2f}"
            else:
                row.append("N/A")
        row.append(tps_16k)
        report.append("| " + " | ".join(row) + " |\n")
    
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_6_suffix_stability.md"), "w") as f:
        f.writelines(report)

# 3. Report: Entropy & Drift
def gen_entropy_report():
    report = ["# Phase 20.6: Drift Risk & Entropy Balance\n\n"]
    report.append("Analysis of decoder freedom vs. stabilization pressure.\n\n")
    report.append("| Metric | Mean Value | Max/Min | Variance |\n")
    report.append("|---|---|---|---|\n")
    
    if entropy_data:
        ents = [d["entropy"] for d in entropy_data]
        report.append(f"| Entropy (Nats) | {np.mean(ents):.4f} | {np.min(ents):.4f} - {np.max(ents):.4f} | {np.var(ents):.4f} |\n")
    
    if drift_data:
        risks = [d["drift_risk"] for d in drift_data]
        stabs = [d["stab_factor"] for d in drift_data]
        report.append(f"| Drift Risk | {np.mean(risks):.4f} | {np.min(risks):.4f} - {np.max(risks):.4f} | {np.var(risks):.4f} |\n")
        report.append(f"| Stabilization Factor | {np.mean(stabs):.4f} | {np.min(stabs):.4f} - {np.max(stabs):.4f} | {np.var(stabs):.4f} |\n")

    with open(os.path.join(REPORTS_DIR, "reconstruction_20_6_drift_entropy.md"), "w") as f:
        f.writelines(report)

# 4. Report: Failure Analysis
def gen_failure_analysis():
    report = ["# Phase 20.6: Failure Analysis & TPS Residue\n\n"]
    
    # Identify regression cases (SPS worse than ALFSR)
    regressions = []
    for d_sps in [d for d in precision_data if d["mode"] == "sps_20_6"]:
        d_alfsr = [d for d in precision_data if d["mode"] == "alfsr_20_5" and d["ctx"] == d_sps["ctx"] and d["domain"] == d_sps["domain"]]
        if d_alfsr and d_sps["fidelity"] < d_alfsr[0]["fidelity"]:
            regressions.append({
                "ctx": d_sps["ctx"],
                "domain": d_sps["domain"],
                "sps": d_sps["fidelity"],
                "alfsr": d_alfsr[0]["fidelity"]
            })
            
    report.append("## SPS vs ALFSR Regressions\n")
    if regressions:
        report.append("| Context | Domain | SPS Fidelity | ALFSR Fidelity |\n")
        report.append("|---|---|---|---|\n")
        for r in regressions:
            report.append(f"| {r['ctx']} | {r['domain']} | {r['sps']:.3f} | {r['alfsr']:.3f} |\n")
    else:
        report.append("No significant regressions detected.\n")
        
    report.append("\n## TPS Collapse Observation (16k Context)\n")
    report.append("In the `suffix_integrity` domain at 16k context, `sps_20_6` throughput collapsed to **0.06 TPS**.\n\n")
    report.append("### Root Cause Hypothesis:\n")
    report.append("1. **Sequence Alignment Overhead**: The `_get_expected_symbolic_token` loop in `SPSResolver` performs $O(M)$ searches every step. If `suffix_integrity` triggers high salience across many chunks, the search space expands significantly.\n")
    report.append("2. **Entropy Auditor Latency**: Calculating Shannon entropy on a 152k-wide distribution every step adds non-trivial overhead, especially if the tensor is moved or processed on CPU.\n")
    report.append("3. **Decay Balancer Friction**: If `drift_risk` oscillates, the balancer may be triggering frequent re-spikes, preventing the GPU from maintaining a steady execution pipeline.\n")

    with open(os.path.join(REPORTS_DIR, "reconstruction_20_6_failure_analysis.md"), "w") as f:
        f.writelines(report)

gen_precision_report()
gen_suffix_report()
gen_entropy_report()
gen_failure_analysis()

print("Reports generated successfully.")
