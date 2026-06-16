import subprocess
import time
import os

# Read the long prompt
with open("scratch/long_prompt.txt", "r") as f:
    prompt = f.read()

# Define test configurations
configs = [
    {
        "name": "GPU + Custom Attn (DIFFKV_NATIVE_ATTN=0)",
        "env": {"DIFFKV_NATIVE_ATTN": "0", "DIFFKV_USE_GPU": "1", "DIFFKV_TIME_DECODE": "1"}
    },
    {
        "name": "GPU + Native Attn (DIFFKV_NATIVE_ATTN=1)",
        "env": {"DIFFKV_NATIVE_ATTN": "1", "DIFFKV_USE_GPU": "1", "DIFFKV_TIME_DECODE": "1"}
    },
    {
        "name": "CPU + Custom Attn (DIFFKV_NATIVE_ATTN=0)",
        "env": {"DIFFKV_NATIVE_ATTN": "0", "DIFFKV_USE_GPU": "0", "DIFFKV_TIME_DECODE": "1"}
    },
    {
        "name": "CPU + Native Attn (DIFFKV_NATIVE_ATTN=1)",
        "env": {"DIFFKV_NATIVE_ATTN": "1", "DIFFKV_USE_GPU": "0", "DIFFKV_TIME_DECODE": "1"}
    }
]

binary_path = "./diffkv_native/build/diffkv_native"
model_path = "./diffkv_native/qwen2.5-0.5b-instruct.gguf"

for config in configs:
    print("\n" + "=" * 80)
    print(f"RUNNING CONFIG: {config['name']}")
    print("=" * 80)
    
    # Merge default env with config env
    env = os.environ.copy()
    env.update(config["env"])
    # Ensure other environment variables are clean
    env["DIFFKV_MICRO_BLOCK_SIZE"] = "64"
    env["DIFFKV_MAX_CONTEXT_SLOTS"] = "512"
    
    # Run binary
    t_start = time.time()
    proc = subprocess.Popen(
        [binary_path, model_path, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # Read output and stderr lines
    stdout_lines = []
    stderr_lines = []
    
    def read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)
            
    def read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)
            
    import threading
    t1 = threading.Thread(target=read_stdout)
    t2 = threading.Thread(target=read_stderr)
    t1.start()
    t2.start()
    
    # Wait with timeout of 60 seconds
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("TIMEOUT EXPIRED (60s) - Terminating process.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            pass
            
    t1.join()
    t2.join()
    
    duration = time.time() - t_start
    print(f"Execution took {duration:.2f} seconds.")
    
    # Print first few and last few lines of stdout
    print("--- STDOUT PREVIEW ---")
    if len(stdout_lines) > 20:
        for line in stdout_lines[:5]:
            print(f"  {line.rstrip()}")
        print("  ...")
        for line in stdout_lines[-10:]:
            print(f"  {line.rstrip()}")
    else:
        for line in stdout_lines:
            print(f"  {line.rstrip()}")
            
    # Print timing lines from stderr
    print("--- TIMING BREAKDOWN (from STDERR) ---")
    timing_lines = [line.rstrip() for line in stderr_lines if "[Timing Step" in line or "Prefill" in line]
    if len(timing_lines) > 20:
        for line in timing_lines[:5]:
            print(f"  {line}")
        print("  ...")
        for line in timing_lines[-10:]:
            print(f"  {line}")
    else:
        for line in timing_lines:
            print(f"  {line}")
            
    # Print any errors in stderr
    error_lines = [line.rstrip() for line in stderr_lines if "error" in line.lower() or "warning" in line.lower() or "fail" in line.lower()]
    if error_lines:
        print("--- INTERESTING STDERR LINES (ERRORS/WARNINGS) ---")
        for line in error_lines[:10]:
            print(f"  {line}")
