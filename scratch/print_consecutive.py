import subprocess
import os

binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"

env = dict(os.environ)
env["DIFFKV_USE_GPU"] = "1"
env["DIFFKV_MICRO_BLOCK_SIZE"] = "64"

p = subprocess.Popen(
    [binary_path, model_path, "-"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False,
    bufsize=0,
    env=env
)

# Read stdout until __READY__
print("Waiting for __READY__...")
buf = b""
while b"__READY__" not in buf:
    c = p.stdout.read(1)
    if not c:
        break
    buf += c
print(f"Subprocess initialized: {buf.decode('utf-8')}")

# Turn 1
prompt1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi, my favorite color is blue.<|im_end|>\n<|im_start|>assistant\n"
single_line1 = prompt1.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
p.stdin.write((single_line1 + "\n").encode('utf-8'))
p.stdin.flush()

print("\n--- Reading Turn 1 ---")
buf1 = b""
while True:
    c = p.stdout.read(1)
    if not c:
        break
    buf1 += c
    # Read until we get __CACHED__ followed by a newline
    if b"__CACHED__:" in buf1 and buf1.endswith(b"\n"):
        break

print(f"Raw Turn 1 output:\n{buf1.decode('utf-8')}")

# Parse cached length
cached_len = 0
for line in buf1.decode('utf-8').split("\n"):
    if line.startswith("__CACHED__:"):
        cached_len = int(line.split(":")[1])
        break

# Extract response
resp_start = buf1.find(b"__RESPONSE__") + len(b"__RESPONSE__\n")
resp_end = buf1.find(b"__FINISH__")
reply = buf1[resp_start:resp_end].decode('utf-8').strip()

# Turn 2
prompt2 = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi, my favorite color is blue.<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n<|im_start|>user\nWhat is my favorite color?<|im_end|>\n<|im_start|>assistant\n"
single_line2 = prompt2.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
stdin_payload = f"__CACHED__:{cached_len}\n{single_line2}\n"

print(f"\nSending Turn 2 (cached_len={cached_len})...")
p.stdin.write(stdin_payload.encode('utf-8'))
p.stdin.flush()

print("\n--- Reading Turn 2 ---")
buf2 = b""
while True:
    c = p.stdout.read(1)
    if not c:
        break
    buf2 += c
    if b"__CACHED__:" in buf2 and buf2.endswith(b"\n"):
        break

print(f"Raw Turn 2 output:\n{buf2.decode('utf-8')}")

p.terminate()
p.wait()
