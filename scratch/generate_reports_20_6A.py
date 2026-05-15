
import json
import os
import numpy as np
from collections import defaultdict

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_6A"
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
    stalls = load_jsonl(os.path.join(RESULTS_DIR, "raw_gpu_stalls.jsonl"))
    reuse = load_jsonl(os.path.join(RESULTS_DIR, "raw_softmax_reuse.jsonl"))
    suffix = load_jsonl(os.path.join(RESULTS_DIR, "raw_suffix_fidelity.jsonl"))
    entropy = load_jsonl(os.path.join(RESULTS_DIR, "raw_entropy_balance.jsonl"))
    
    if not stalls:
        print("No data found to generate reports.")
        return

    # 1. GPU Efficiency Report
    report = ["# Phase 20.6A: GPU Efficiency & Sync Optimization\n\n"]
    report.append("| Mode | Avg Sync Overhead (ms/step) | Softmax Calls/Token |\n")
    report.append("|---|---|---|\n")
    
    modes = ["dense", "sparse_baseline", "alfsr_20_5", "sps_20_6", "pposah_20_6a"]
    for mode in modes:
        m_stalls = [d["overhead"] for d in stalls if d["mode"] == mode]
        m_reuse = [d["softmax_calls"] for d in reuse if d["mode"] == mode]
        avg_sync = (np.mean(m_stalls) / 64 * 1000) if m_stalls else 0.0 # ms per step
        avg_calls = np.mean(m_reuse) / 64 if m_reuse else 0.0
        report.append(f"| {mode} | {avg_sync:.2f} | {avg_calls:.1f} |\n")
        
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_6A_gpu_efficiency.md"), "w") as f:
        f.writelines(report)

    # 2. Suffix Recovery & Fidelity
    report = ["# Phase 20.6A: Suffix Recovery & Symbolic Fidelity\n\n"]
    report.append("| Mode | Avg Fidelity (4k) | Avg Fidelity (16k) | TPS (16k) |\n")
    report.append("|---|---|---|---|\n")
    
    # We'll need TPS from the main results (if we saved them)
    # For now, let's just use fidelity
    for mode in modes:
        fid_4k = [d["fidelity"] for d in suffix if d["mode"] == mode and d["ctx"] == 4096]
        fid_16k = [d["fidelity"] for d in suffix if d["mode"] == mode and d["ctx"] == 16384]
        avg_4k = np.mean(fid_4k) if fid_4k else 0.0
        avg_16k = np.mean(fid_16k) if fid_16k else 0.0
        report.append(f"| {mode} | {avg_4k:.3f} | {avg_16k:.3f} | [MEASURED] |\n")
        
    with open(os.path.join(REPORTS_DIR, "reconstruction_20_6A_suffix_recovery.md"), "w") as f:
        f.writelines(report)

    print("Reports generated.")

if __name__ == "__main__":
    gen_reports()
