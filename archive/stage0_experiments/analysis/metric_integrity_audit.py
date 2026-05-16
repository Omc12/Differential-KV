import os
import sys
import json
import torch
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def audit_results(input_dir, output_path):
    print(f"\n>>> Auditing Metric Integrity in {input_dir}")
    audit_report = {"files_checked": [], "anomalies": [], "integrity_score": 1.0}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file); audit_report["files_checked"].append(file)
                try:
                    with open(file_path, "r") as f: data = json.load(f)
                    def check_recursive(d, path=""):
                        if isinstance(d, dict):
                            for k, v in d.items(): check_recursive(v, f"{path}/{k}")
                        elif isinstance(d, list):
                            for i, v in enumerate(d): check_recursive(v, f"{path}[{i}]")
                        elif isinstance(d, float):
                            if np.isnan(d) or np.isinf(d): audit_report["anomalies"].append(f"{file_path} {path}: {d}")
                            if d < 0 and "ratio" not in path.lower() and "acc" not in path.lower(): audit_report["anomalies"].append(f"{file_path} {path}: Negative value {d}")
                    check_recursive(data)
                except Exception as e: audit_report["anomalies"].append(f"Error reading {file}: {e}")
    if audit_report["anomalies"]: audit_report["integrity_score"] = max(0, 1.0 - len(audit_report["anomalies"]) / 100.0); print(f"  Found {len(audit_report['anomalies'])} anomalies!")
    else: print("  All metrics passed integrity audit.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f: json.dump(audit_report, f, indent=4)
    print(f"Audit report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/phase20")
    parser.add_argument("--output", type=str, default="results/phase20/integrity_audit.json")
    args = parser.parse_args()
    audit_results(args.input_dir, args.output)
