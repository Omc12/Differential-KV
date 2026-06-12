import subprocess
import time
import sys
import os

binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"

# Setup environment variables
env = dict(os.environ)
env["DIFFKV_USE_GPU"] = "1"
env["DIFFKV_MICRO_BLOCK_SIZE"] = "64"

print("Starting subprocess...")
p = subprocess.Popen(
    [binary_path, model_path, "-"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False,
    bufsize=0,
    env=env
)

# Read stderr in background and write to file
import threading
f_err = open("scratch/stderr_reuse.log", "wb")
def read_stderr():
    for line in p.stderr:
        f_err.write(line)
        f_err.flush()

threading.Thread(target=read_stderr, daemon=True).start()

# Read stdout until __READY__
buffer = b""
while True:
    b = p.stdout.read(1)
    if not b:
        break
    buffer += b
    if b"__READY__" in buffer:
        break

# Turn 1
prompt1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi, my favorite color is blue.<|im_end|>\n<|im_start|>assistant\n"
single_line1 = prompt1.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
p.stdin.write((single_line1 + "\n").encode('utf-8'))
p.stdin.flush()

buffer = b""
while True:
    b = p.stdout.read(1)
    if not b:
        break
    buffer += b
    if b"__FINISH__" in buffer and b"\n" in buffer[buffer.find(b"__FINISH__"):]:
        remainder = b""
        for _ in range(50):
            c = p.stdout.read(1)
            remainder += c
            if b"\n" in remainder and b"__CACHED__:" in remainder:
                break
        buffer += remainder
        break

# Parse cached length
cached_len = 0
buffer_str = buffer.decode('utf-8', errors='replace')
for line in buffer_str.split("\n"):
    if line.startswith("__CACHED__:"):
        cached_len = int(line.split(":")[1])

# Extract response text to build Turn 2 prompt
resp_start = buffer_str.find("__RESPONSE__") + len("__RESPONSE__\n")
resp_end = buffer_str.find("__FINISH__")
reply = buffer_str[resp_start:resp_end].strip()

# Turn 2 with cached_len prefix-reuse
prompt2 = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi, my favorite color is blue.<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n<|im_start|>user\nWhat is my favorite color?<|im_end|>\n<|im_start|>assistant\n"
single_line2 = prompt2.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
stdin_payload = f"__CACHED__:{cached_len}\n{single_line2}\n"

p.stdin.write(stdin_payload.encode('utf-8'))
p.stdin.flush()

buffer2_reuse = b""
while True:
    b = p.stdout.read(1)
    if not b:
        break
    buffer2_reuse += b
    if b"__FINISH__" in buffer2_reuse:
        break

p.terminate()
p.wait()

# Extract reuse output
str_reuse = buffer2_reuse.decode('utf-8', errors='replace')
start_idx = str_reuse.find("__RESPONSE__") + len("__RESPONSE__\n")
end_idx = str_reuse.find("__FINISH__")
output_reuse = str_reuse[start_idx:end_idx].strip()

# Start a fresh process for Turn 2 with cached_len=0 (baseline)
print("\nStarting fresh subprocess for baseline (cached_len=0)...")
p_base = subprocess.Popen(
    [binary_path, model_path, "-"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False,
    bufsize=0,
    env=env
)

f_base_err = open("scratch/stderr_base.log", "wb")
def read_base_stderr():
    for line in p_base.stderr:
        f_base_err.write(line)
        f_base_err.flush()

threading.Thread(target=read_base_stderr, daemon=True).start()

buffer = b""
while True:
    b = p_base.stdout.read(1)
    if not b:
        break
    buffer += b
    if b"__READY__" in buffer:
        break

# Send prompt2 with no cache prefix
stdin_payload_base = single_line2 + "\n"
p_base.stdin.write(stdin_payload_base.encode('utf-8'))
p_base.stdin.flush()

buffer2_base = b""
while True:
    b = p_base.stdout.read(1)
    if not b:
        break
    buffer2_base += b
    if b"__FINISH__" in buffer2_base:
        break

p_base.terminate()
p_base.wait()

# Extract baseline output
str_base = buffer2_base.decode('utf-8', errors='replace')
start_idx = str_base.find("__RESPONSE__") + len("__RESPONSE__\n")
end_idx = str_base.find("__FINISH__")
output_base = str_base[start_idx:end_idx].strip()

print("\n==================================================")
print("COMPARISON RESULTS:")
print("==================================================")
print(f"Prefix Reuse Output (cached_len={cached_len}):")
print(f"  {output_reuse!r}")
print("\nBaseline Output (cached_len=0):")
print(f"  {output_base!r}")
print("==================================================")
if output_reuse == output_base:
    print("MATCH! Prefix reuse behaves identically to full prefill baseline!")
else:
    print("MISMATCH! Check prefix reuse decompression / prior context implementation.")
print("==================================================")
