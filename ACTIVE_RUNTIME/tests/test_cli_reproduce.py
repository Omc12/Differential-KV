import subprocess
import os
import sys

# Prepare input text
with open("ACTIVE_RUNTIME/nat_paper_with_needle.txt", "r") as f:
    paper = f.read()

question = (
    "Use only the supplied document.\n\n"
    "What verification tag was assigned to the neighborhood attention analysis?\n\n"
    "Output only the verification tag.\n\n"
    "Do not explain your reasoning.\n\n"
    "Do not output any additional words."
)

# In raw paste mode, we first type /paste, then write the text. The input stream will close at the end.
input_data = f"/paste\n{paper}\n\n{question}\n"

env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
env["PYTHONPATH"] = "ACTIVE_RUNTIME"

p = subprocess.Popen(
    ["./diffkv_venv/bin/python3", "ACTIVE_RUNTIME/serving/cli.py",
     "--model", "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
     "--preset", "mid",
     "--serving-mode", "performance"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env
)

try:
    stdout, stderr = p.communicate(input=input_data, timeout=180)
    print("=== CLI STDOUT ===")
    print(stdout)
    print("=== CLI STDERR ===")
    print(stderr)
except subprocess.TimeoutExpired:
    p.kill()
    print("CLI TIMEOUT!")
