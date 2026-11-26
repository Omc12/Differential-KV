"""Phase 1 gate: DKV's CUSTOM backend is token-identical to stock vLLM.

    python integrations/vllm/tests/phase1_identical.py --model Qwen/Qwen2.5-1.5B-Instruct

Runs the same greedy prompts through stock vLLM (TRITON_ATTN, the backend
phase 1 delegates to) and through the CUSTOM backend, each in its own process
so the two engines never share GPU memory, and compares the generated token
ids exactly. The CUSTOM run must also print the backend's marker line, so a
silent fallback to another backend cannot pass.
"""
import argparse
import json
import subprocess
import sys

MARKER = "[dkv_vllm] DKV attention backend active"
PROMPTS = [
    "The capital of France is",
    "Write a haiku about rivers.",
    "List three prime numbers greater than 100:",
    "Explain what a KV cache is in one sentence.",
    "def fibonacci(n):",
    "Translate to German: The weather is nice today.",
    "Q: What is 17 * 23? A:",
    "Summarize the plot of Hamlet in two sentences.",
    "The three primary colors are",
    "In 1969, humans first",
] * 2


def run(backend, model, max_tokens, out):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, attention_backend=backend, gpu_memory_utilization=0.55,
              max_model_len=4096, enforce_eager=True, seed=0)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    json.dump({"backend": backend,
               "tokens": [list(o.outputs[0].token_ids) for o in outs]}, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--role", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.role:
        return run(a.role, a.model, a.max_tokens, a.out)
    res, logs = {}, {}
    for b in ("TRITON_ATTN", "CUSTOM"):
        out = f"/tmp/dkv_vllm_phase1_{b}.json"
        p = subprocess.run([sys.executable, __file__, "--role", b, "--model", a.model,
                            "--max-tokens", str(a.max_tokens), "--out", out],
                           capture_output=True, text=True)
        logs[b] = p.stdout + p.stderr
        if p.returncode:
            print(logs[b][-4000:])
            sys.exit(f"{b} run failed")
        res[b] = json.load(open(out))
    t0, t1 = res["TRITON_ATTN"]["tokens"], res["CUSTOM"]["tokens"]
    same = sum(x == y for x, y in zip(t0, t1))
    used = MARKER in logs["CUSTOM"]
    print(f"DKV backend used in the CUSTOM run: {used}; "
          f"stock run printed the marker: {MARKER in logs['TRITON_ATTN']}")
    print(f"token-identical prompts: {same}/{len(t0)}")
    ok = used and MARKER not in logs["TRITON_ATTN"] and same == len(t0) == len(t1)
    print("PHASE 1 GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
