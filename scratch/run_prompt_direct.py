import os
import sys
import subprocess
import threading
import time

def read_stderr_thread(proc):
    try:
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            print(f"[C++ stderr] {line.strip()}", flush=True)
    except Exception as e:
        print(f"stderr read error: {e}", flush=True)

def read_until(proc, token):
    buf = b""
    while token not in buf:
        try:
            c = os.read(proc.stdout.fileno(), 4096)
            if not c:
                break
            buf += c
            # Print chunks as we receive them to keep logs alive
            sys.stdout.write(c.decode("utf-8", "replace"))
            sys.stdout.flush()
        except Exception as e:
            print(f"\nRead error: {e}", flush=True)
            break
    return buf

def main():
    prompt_file = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/prompt_4096.txt"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "diffkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    
    with open(prompt_file, "r") as f:
        prompt_text = f.read()
        
    # Escape newlines and backslashes
    escaped = prompt_text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
    
    binary_path = "./diffkv_native/build/diffkv_native"
    
    env = os.environ.copy()
    env["DIFFKV_ENABLE_FACTUAL"] = os.environ.get("DIFFKV_ENABLE_FACTUAL", "0")
    env["DIFFKV_VERBOSE"] = "1"
    env["DIFFKV_MAX_TOKENS"] = "128"
    env["DIFFKV_TEMPERATURE"] = "0.0"  # greedy
    
    proc = subprocess.Popen(
        [binary_path, model_path, "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env
    )
    
    # Start thread to read stderr line by line
    stderr_thread = threading.Thread(target=read_stderr_thread, args=(proc,), daemon=True)
    stderr_thread.start()
    
    print("[Python] Waiting for __READY__...", flush=True)
    read_until(proc, b"__READY__")
    print("\n[Python] Sending prompt...", flush=True)
    
    proc.stdin.write((escaped + "\n").encode("utf-8"))
    proc.stdin.flush()
    
    print("[Python] Waiting for __RESPONSE__...", flush=True)
    read_until(proc, b"__RESPONSE__")
    
    print("\n[Python] Reading response...", flush=True)
    read_until(proc, b"__FINISH__")
    
    print("\n[Python] Done.", flush=True)
    try:
        proc.stdin.write(b"exit\n")
        proc.stdin.flush()
    except Exception:
        pass
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    main()
