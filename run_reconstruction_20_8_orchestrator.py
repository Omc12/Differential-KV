
import subprocess
import os
import json
import time

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_8"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_case(mode, ctx, domain, prop_len):
    cmd = [
        "python", 
        "d:\\Codes\\Projects\\Differential KV\\scratch\\single_run_20_8.py",
        "--mode", mode,
        "--ctx", str(ctx),
        "--domain", domain,
        "--prop_len", str(prop_len)
    ]
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Launching: {mode} {ctx} {domain} {prop_len}")
        subprocess.run(cmd, check=True, timeout=300)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Failed: {e}")

if __name__ == "__main__":
    modes = ["dense", "sparse_baseline", "pposah_20_6a", "spslrif_20_7", "sabeaf_20_8"]
    contexts = [4096, 8192, 16384]
    domains = [
        "hex_sequence", "api_key_complex", "activation_code", 
        "json_reconstruction", "adversarial_delimiters", "anchor_fragmentation"
    ]
    prop_lengths = [64, 128]

    for ctx in contexts:
        for prop_len in prop_lengths:
            for domain in domains:
                for mode in modes:
                    run_case(mode, ctx, domain, prop_len)
