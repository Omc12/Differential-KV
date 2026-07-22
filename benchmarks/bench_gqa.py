#!/usr/bin/env python3
import subprocess
import os
import re
import sys
import time

# Define base sweep env
env_base = {
    "DKV_ENGAGE_THRESHOLD": "1024",
    "DKV_NATIVE_ATTN": "0",  # default path (triggers execute_cpu_attention)
    "DKV_FORCE_CPU_ATTN": "1",  # force CPU to print LSE2
    "DKV_MPS_APPROXIMATE_ATTN": "1",
    "DKV_DENSE_DIRECT": "1",
    "DKV_POOL_ABS_ROT": "1",
    "DKV_TEMPERATURE": "0",
    "DKV_DISABLE_VSL": "1",
    "DKV_ENABLE_FACTUAL": "0",
    "DKV_MAX_TOKENS": "40",
    "DKV_REPETITION_PENALTY": "1.0",
    "DKV_DBG_LSE2": "1",
    "HF_HUB_OFFLINE": "1"
}

binary = "../dkv_native/build/dkv_native"
model = "../dkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
make_prompt_script = "../dkv_native/tests/make_niah_prompt.py"

needle = "The secret passcode is OMEGA-7741-DELTA."
question = "What is the secret passcode? Repeat it exactly."

def run_cell(ctx, depth, gqa_on):
    env = os.environ.copy()
    env.update(env_base)
    env["DKV_CB_GQA_ROUTE"] = "1" if gqa_on else "0"
    
    # Generate prompt
    cmd_prompt = [
        sys.executable,
        make_prompt_script,
        str(ctx),
        str(depth),
        needle,
        question
    ]
    prompt = subprocess.check_output(cmd_prompt).decode("utf-8", errors="replace").strip()
    
    # Run binary
    cmd_bin = [binary, model, prompt]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd_bin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    stdout, stderr = proc.communicate()
    dt = time.perf_counter() - t0
    
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    
    # Check if passed
    passed = "OMEGA-7741-DELTA" in stdout_text or "omega-7741-delta" in stdout_text.lower()
    
    # Calculate decode TPS
    # We generated max 40 tokens (plus prompt prefill). Decode takes exactly max_tokens (40).
    tps = 40.0 / dt if dt > 0 else 0.0
    
    # Extract margin from step 7 predictions dump (where it usually outputs "-")
    margin = None
    steps = stderr_text.split("\n[Step ")
    for step_block in steps[1:]:
        lines = step_block.split("\n")
        step_header = lines[0]
        if len(lines) >= 3:
            top_line = lines[1]
            if "0: " in top_line and "id: 12" in top_line:
                m1 = re.search(r"logit:\s*([-\d\.]+)", top_line)
                m2 = re.search(r"logit:\s*([-\d\.]+)", lines[2])
                if m1 and m2:
                    margin = float(m1.group(1)) - float(m2.group(1))
                    break
                    
    # Also extract routing times from logs if available
    # Actually, we timed the overall run which includes prefill, but decode tps is dominated by decode steps.
    return passed, tps, margin, stdout_text.strip()

print("=====================================================")
# We run 2 runs each to account for warmup/noise
print("GQA Routing ON vs OFF Benchmarking at 16k context")
print("=====================================================")

for depth in [0.5, 0.9]:
    for gqa_on in [True, False]:
        gqa_str = "ON" if gqa_on else "OFF"
        print(f"\n--- Depth {depth} | GQA Route {gqa_str} ---")
        
        runs = []
        for run_id in range(2):
            passed, tps, margin, out = run_cell(16000, depth, gqa_on)
            pass_str = "PASS" if passed else "FAIL"
            margin_str = f"{margin:.4f}" if margin is not None else "N/A"
            print(f"  Run {run_id+1}: {pass_str} | TPS: {tps:.2f} | Margin: {margin_str} | Output: {out[:50]}")
            runs.append((passed, tps, margin))
