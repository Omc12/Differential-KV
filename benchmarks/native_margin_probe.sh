#!/usr/bin/env python3
import subprocess
import os
import re
import sys

# Define standard sweep env
env = os.environ.copy()
env["DIFFKV_ENGAGE_THRESHOLD"] = "1024"
# Probe the DEFAULT decode path (NATIVE_ATTN=0), the same path as the honest 6-cell
# sweep and what users run. NATIVE_ATTN=1 (the fused-ggml path W7 removed) does NOT
# degrade gracefully — it emits gibberish at 16k ("The secret secretTheThe unary…"),
# which made this probe report a false 16k failure. Fixed 2026-07-04.
env["DIFFKV_NATIVE_ATTN"] = "0"
env["DIFFKV_FORCE_CPU_ATTN"] = "0"
env["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
env["DIFFKV_DENSE_DIRECT"] = "1"
env["DIFFKV_POOL_ABS_ROT"] = "1"
env["DIFFKV_TEMPERATURE"] = "0"
env["DIFFKV_DISABLE_VSL"] = "1"
env["DIFFKV_ENABLE_FACTUAL"] = "0"
env["DIFFKV_REPETITION_PENALTY"] = "1.0"
env["DIFFKV_MAX_TOKENS"] = "20"
env["HF_HUB_OFFLINE"] = "1"

binary = "../diffkv_native/build/diffkv_native"
model = "../diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
make_prompt_script = "../diffkv_native/tests/make_niah_prompt.py"

needle = "The secret passcode is OMEGA-7741-DELTA."
question = "What is the secret passcode? Repeat it exactly."

def run_probe(ctx):
    print(f"\n--- Running probe for ctx={ctx} ---")
    # Generate prompt
    cmd_prompt = [
        sys.executable,
        make_prompt_script,
        str(ctx),
        "0.5",
        needle,
        question
    ]
    prompt = subprocess.check_output(cmd_prompt).decode("utf-8", errors="replace").strip()
    
    # Run binary
    cmd_bin = [binary, model, prompt]
    proc = subprocess.Popen(
        cmd_bin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    stdout, stderr = proc.communicate()
    stderr_text = stderr.decode("utf-8", errors="replace")
    stdout_text = stdout.decode("utf-8", errors="replace")
    
    # Parse stderr for Step predictions
    # We want the step where top-1 is "-" (id: 12)
    # We parse blocks of the form:
    # [Step X Top predictions]:
    #   0: "-" (id: 12, logit: 14.5)
    #   1: ...
    
    # Let's split by "\n[Step "
    steps = stderr_text.split("\n[Step ")
    for step_block in steps[1:]:
        # Re-construct step header and content
        lines = step_block.split("\n")
        step_header = lines[0] # e.g. "5 Top predictions]:"
        
        # Check if this block contains id: 12 as the top prediction
        if len(lines) >= 3:
            # line 1 should start with "  0: " and contain "id: 12"
            top_line = lines[1]
            if "0: " in top_line and "id: 12" in top_line:
                # Top-1 is indeed id: 12
                # Let's extract logit for top-1 and top-2
                m1 = re.search(r"logit:\s*([-\d\.]+)", top_line)
                m2 = re.search(r"logit:\s*([-\d\.]+)", lines[2])
                if m1 and m2:
                    l1 = float(m1.group(1))
                    l2 = float(m2.group(1))
                    margin = l1 - l2
                    print(f"Detected top-1 token id 12 (-) at Step {step_header.split()[0]}")
                    print(f"  Top-1 logit: {l1:.4f}")
                    print(f"  Top-2 logit: {l2:.4f}")
                    print(f"  Margin: {margin:.4f}")
                    return margin
    print("WARNING: Could not detect step where top-1 was token id 12 (-)")
    print("STDOUT was:")
    print(stdout_text)
    return None

print("=====================================================")
print("MARGIN-BASED GUARDRAIL PROBE")
print("=====================================================")
m8k = run_probe(8000)
m16k = run_probe(16000)

print("\n=====================================================")
print("SUMMARY MARGINS")
print("=====================================================")
if m8k is not None:
    print(f"8k / 0.5 margin:  {m8k:.4f}")
if m16k is not None:
    print(f"16k / 0.5 margin: {m16k:.4f}")
