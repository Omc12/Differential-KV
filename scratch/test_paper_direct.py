import subprocess
import time
import os

with open("/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/scratch/random_features_paper.txt", "r") as f:
    paper_text = f.read()

prompt = paper_text + "\n\nBased on the text above, summarize the key contributions and features in a detailed bulleted list:"
chat_prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n" + prompt + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# We will run both GPU attention (Metal) and CPU attention
for force_cpu in ["0", "1"]:
    print(f"\n==========================================")
    print(f"Running C++ binary with DIFFKV_FORCE_CPU_ATTN={force_cpu}")
    print(f"==========================================")
    
    start = time.time()
    res = subprocess.run(
        ["./diffkv_native/build/diffkv_native", "./diffkv_native/qwen2.5-0.5b-instruct.gguf", chat_prompt],
        capture_output=True,
        text=False,
        env={
            "DIFFKV_FORCE_CPU_ATTN": force_cpu,
            "DIFFKV_USE_GPU": "1",
            "DIFFKV_MICRO_BLOCK_SIZE": "64",
            "DIFFKV_MAX_TOKENS": "512", # restrict generated tokens to 512
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
        }
    )
    dur = time.time() - start
    print(f"Finished in {dur:.2f} seconds.")
    print("=== STDOUT ===")
    print(res.stdout.decode("utf-8", errors="replace"))
    print("=== STDERR ===")
    # Print only the last 30 lines of stderr to keep it readable, or print all if short
    stderr_lines = res.stderr.decode("utf-8", errors="replace").splitlines()
    print("\n".join(stderr_lines[-40:]))
