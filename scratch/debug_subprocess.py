import subprocess
import time
import sys

binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"

print("Starting subprocess...")
p = subprocess.Popen(
    [binary_path, model_path, "-"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Read stderr in a non-blocking way or print it
import threading
def read_stderr():
    for line in p.stderr:
        print(f"[C++ STDERR] {line.strip()}", flush=True)

threading.Thread(target=read_stderr, daemon=True).start()

# Read stdout until __READY__
print("Waiting for __READY__...")
buffer = ""
while True:
    char = p.stdout.read(1)
    if not char:
        print("Subprocess exited unexpectedly during startup")
        sys.exit(1)
    buffer += char
    if "__READY__" in buffer:
        print(f"[C++ STDOUT] {buffer.strip()}")
        break

# Send a prompt
prompt = "Question: What is the secret code word? Answer in exactly one word."
print(f"Sending prompt: {prompt}")
p.stdin.write(prompt + "\n")
p.stdin.flush()

print("Reading response...")
buffer = ""
t0 = time.time()
while True:
    char = p.stdout.read(1)
    if not char:
        print("Subprocess exited unexpectedly during generation")
        break
    sys.stdout.write(char)
    sys.stdout.flush()
    buffer += char
    if "__FINISH__" in buffer:
        print("\n__FINISH__ received!")
        break
    if time.time() - t0 > 30:
        print("\nTimeout of 30s reached. Exiting.")
        break

p.terminate()
p.wait()
