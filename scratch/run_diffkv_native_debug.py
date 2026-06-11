import subprocess
import os
import sys

def main():
    # Force DiffKV engage
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
    os.environ["DIFFKV_IMMEDIATE_PREFILL_COMPRESS"] = "1"
    
    binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
    model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
    
    with open("/Users/omchimurkar1/Desktop/Differential-KV/scratch/debug_prompt.txt", "r") as f:
        prompt = f.read()
        
    print("Running diffkv_native in non-interactive mode...")
    cmd = [binary_path, model_path, prompt]
    
    # We run the command and stream output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout_lines = []
    stderr_lines = []
    
    # We want to read both stdout and stderr in real-time or just wait and print
    # Let's read them
    stdout, stderr = proc.communicate()
    
    print("\n--- STDERR ---")
    print(stderr)
    print("\n--- STDOUT ---")
    print(stdout)

if __name__ == "__main__":
    main()
