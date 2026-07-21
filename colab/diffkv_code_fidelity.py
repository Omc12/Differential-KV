#!/usr/bin/env python3
"""A100 validation for the random-code fidelity fix (Rule 1b in
streaming_sparse_ingest._should_skip_compression).

Runs the EXACT harness codes (run_a100_paper_experiments.generate_random_needles,
seed 42, PREFIX-{1000..9999}-SUFFIX) through the PRODUCTION CUDA DiffKV path at
8k / depth 0.5 and reports per-code recall + which skip_compression rule fired.

Baseline (before Rule 1b): ~33% — the 4-digit codes fail the \\d{5,} gate, get
lossy SVD, decode to garbage. MLX gets 100% (no gate). Expected after the fix:
~100% (codes hit Rule 1b -> skip_compression -> exact residuals).

Run (transformers MUST be 4.46.3):
  DIFFKV_TELEMETRY=1 python colab/diffkv_code_fidelity.py --model Qwen/Qwen2.5-0.5B-Instruct
  # DIFFKV_TELEMETRY=1 prints "[DiffKV DEBUG] Rule 1b skip block ..." for the needle block.
"""
import os, sys, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
import torch

_PREFIXES = ["OMEGA","SIGMA","THETA","LAMBDA","KAPPA","NEXUS","CYPHER","VORTEX","APEX","TITAN"]
_SUFFIXES = ["DELTA","BETA","ALPHA","GAMMA","ZETA","PRIME","MATRIX","VECTOR","SHIELD","ORBIT"]

def rand_codes(count):
    rng = random.Random(42)
    out, seen = [], set()
    while len(out) < count:
        c = f"{rng.choice(_PREFIXES)}-{rng.randint(1000,9999)}-{rng.choice(_SUFFIXES)}"
        if c not in seen:
            seen.add(c); out.append(c)
    return out

FILLER = ("Quantum computing is a multidisciplinary field comprising aspects of computer science, "
          "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
          "than on classical computers. The field of quantum computing includes hardware research and "
          "application development. ")

def make_prompt(tok, ctx, depth, needle, question):
    ft = tok.encode(FILLER, add_special_tokens=False)
    nt = tok.encode(needle + "\n", add_special_tokens=False)
    budget = max(100, ctx - len(nt) - 100)
    allf = (ft * ((budget // len(ft)) + 1))[:budget]
    ins = int(len(allf) * depth)
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            + tok.decode(allf[:ins]) + "\n" + needle + "\n" + tok.decode(allf[ins:])
            + "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    from serving.decode_config import apply_best_decode_defaults
    apply_best_decode_defaults()                       # SPARSE_BIAS=auto etc. — production path
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codes = rand_codes(args.n)
    print(f"[CODE-FIDELITY] model={args.model} ctx={args.ctx} depth={args.depth} n={args.n}\n  codes={codes}\n")

    passes = 0
    for code in codes:
        w = DiffKVHFWrapper(args.model, config={"rank": 32}, device=device)
        needle = f"The secret security passcode is {code}."
        question = "What is the secret security passcode? Repeat it exactly."
        prompt = make_prompt(w.tokenizer, args.ctx, args.depth, needle, question)
        ptoks = len(w.tokenizer.encode(prompt))
        w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        sid = w.active_session or "default"
        gt = w.tokenizer.decode(w._session_token_ids.get(sid, [])[ptoks:], skip_special_tokens=True)
        hit = code in gt
        passes += int(hit)
        print(f"  [{'PASS' if hit else 'FAIL'}] {code:<20} plen={ptoks} gen={gt[:60]!r}", flush=True)
        w.stop()
    print(f"\n[CODE-FIDELITY] CUDA recall: {passes}/{args.n} = {100*passes/args.n:.0f}%  "
          f"(pre-fix ~33%, MLX 100%; expect ~100% after Rule 1b)")

if __name__ == "__main__":
    main()
