import os
import sys
import time
import subprocess

def run_profile():
    binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
    model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
    
    with open("scratch/wiki_prompt.txt", "r") as f:
        prompt = f.read()
    
    env = os.environ.copy()
    env["DIFFKV_USE_GPU"] = "1"
    env["DIFFKV_VERBOSE"] = "1"
    env["DIFFKV_TIME_DECODE"] = "1"
    env["DIFFKV_MAX_TOKENS"] = "5" # profile 5 steps to see the pattern
    env["DIFFKV_PRESET"] = "low"
    
    cmd = [binary_path, model_path, prompt]
    
    print("Launching C++ binary on 10.7k token prompt...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    stdout_lines, stderr_lines = [], []
    
    def read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)
            
    def read_stderr():
        for line in proc.stderr:
            print(f"  [stderr] {line.strip()}")
            stderr_lines.append(line)
            
    import threading
    t_out = threading.Thread(target=read_stdout)
    t_err = threading.Thread(target=read_stderr)
    t_out.start()
    t_err.start()
    
    proc.wait()
    t_out.join()
    t_err.join()
    
    print("Done!")

if __name__ == "__main__":
    run_profile()
