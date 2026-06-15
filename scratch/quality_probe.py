#!/usr/bin/env python3
"""
Quality probe — mirrors ACTIVE_RUNTIME tests/test_niah.py exactly (chat template,
6-digit needle at a depth), run through the diffkv_native C++ binary. Sparse vs
dense, and a rank sweep, to isolate the sparse-mode quality regression.
"""
import os, subprocess, time

BIN = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
MODEL = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
NEEDLE = "The special code is 847291."
QUESTION = "What is the special code? Answer in exactly the 6-digit code number."

# Build a NIAH prompt like test_niah.make_niah_prompt (filler repeated, needle at depth).
filler = ("Quantum computing is a multidisciplinary field comprising aspects of computer science, "
          "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
          "than on classical computers. The field of quantum computing includes hardware research and "
          "application development. Quantum computers are able to solve certain classes of problems "
          "much faster than classical computers by taking advantage of quantum mechanical effects, "
          "such as superposition and quantum entanglement. ")

def build(depth=0.5, repeats=42):
    body = filler * repeats
    ins = int(len(body) * depth)
    haystack = body[:ins] + "\n" + NEEDLE + "\n" + body[ins:]
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + haystack + "\n\n" + QUESTION + "<|im_end|>\n"
            "<|im_start|>assistant\n")

def run(prompt, extra_env):
    env = dict(os.environ)
    env.update({"DIFFKV_USE_GPU": "1", "DIFFKV_TEMPERATURE": "0.01", "DIFFKV_TOP_P": "1.0",
                "DIFFKV_SEED": "123", "DIFFKV_MAX_TOKENS": "16"})
    env.update(extra_env)
    line = prompt.replace("\n", "\\n")
    t0 = time.time()
    p = subprocess.run([BIN, MODEL, "-"], input=line + "\nexit\n",
                       capture_output=True, text=True, env=env, timeout=900)
    dt = time.time() - t0
    ans = []
    for chunk in p.stdout.split("__RESPONSE__")[1:]:
        seg = chunk.split("__FINISH__")[0].replace("[Response]", "").strip()
        if seg: ans.append(seg)
    return (" | ".join(ans) if ans else "(none)"), dt

if __name__ == "__main__":
    prompt = build(depth=0.5, repeats=30)
    print(f"[probe] chat-templated NIAH, words≈{len(prompt.split())}, needle=847291 @ depth 0.5\n")
    configs = [
        ("DENSE  rank16", {"DIFFKV_ENGAGE_THRESHOLD": "6144", "DIFFKV_RANK": "16"}),
        ("SPARSE rank16", {"DIFFKV_ENGAGE_THRESHOLD": "2048", "DIFFKV_RANK": "16"}),
        ("SPARSE rank32", {"DIFFKV_ENGAGE_THRESHOLD": "2048", "DIFFKV_RANK": "32"}),
        ("SPARSE rank64", {"DIFFKV_ENGAGE_THRESHOLD": "2048", "DIFFKV_RANK": "64"}),
    ]
    for name, env in configs:
        ans, dt = run(prompt, env)
        ok = "847291" in ans
        print(f"{name}: [{dt:4.1f}s] {'PASS' if ok else 'FAIL'}  ans={ans!r}")
