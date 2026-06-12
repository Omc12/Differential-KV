import subprocess
import time

with open("scratch/full_paper_prompt.txt", "r") as f:
    paper_text = f.read()

prompt = paper_text + "\nBased on the text above, summarize the key contribution of the paper in one sentence:"

for force_cpu in ["0", "1"]:
    print(f"\n==========================================")
    print(f"Running C++ binary with DIFFKV_FORCE_CPU_ATTN={force_cpu}")
    print(f"==========================================")
    
    start = time.time()
    res = subprocess.run(
        ["./diffkv_native/build/diffkv_native", "diffkv_native/qwen2.5-0.5b-instruct.gguf", prompt],
        capture_output=True,
        text=False,
        env={"DIFFKV_FORCE_CPU_ATTN": force_cpu, "DIFFKV_USE_GPU": "1", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"}
    )
    dur = time.time() - start
    
    stdout_str = res.stdout.decode('utf-8', errors='replace')
    stderr_str = res.stderr.decode('utf-8', errors='replace')
    
    print(f"\nFinished in {dur:.2f} seconds.")
    print("=== STDOUT ===")
    print(stdout_str)
    print("=== STDERR ===")
    print(stderr_str)
