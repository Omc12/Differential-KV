"""Phases 2-3 gate: DKV codec pages and routed decode inside vLLM.

    python integrations/vllm/tests/phase23_gate.py --model Qwen/Qwen2.5-1.5B-Instruct

Arms, each in its own process (one engine on the GPU at a time):
  stock     vLLM's Triton backend, 16-bit KV            -- the reference
  tq_k8v4   vLLM's own TurboQuant k8v4 KV cache          -- equal-bytes in-tree baseline
  dkv_all   DKV k8v4 codec pages, every page read        -- phase 2
  dkv_route DKV k8v4 codec pages, routed decode          -- phase 3

Measured per arm:
  short     greedy agreement with stock on 20 short prompts (exact match and
            mean matching-prefix length)
  needle    passkey retrieval at ~CTX tokens, 6 depths (exact digits)
  capacity  KV cache tokens vLLM reports at startup ("GPU KV cache size")
  decode    tokens/s generating 128 tokens after a CTX-token prompt
  routing   share of pages a routed decode step read (dkv_route)
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

SHORT = [
    "The capital of France is", "Write a haiku about rivers.",
    "List three prime numbers greater than 100:",
    "Explain what a KV cache is in one sentence.", "def fibonacci(n):",
    "Translate to German: The weather is nice today.", "Q: What is 17 * 23? A:",
    "Summarize the plot of Hamlet in two sentences.", "The three primary colors are",
    "In 1969, humans first",
] * 2
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
          "There and back again. ")

ARMS = {
    "stock": {"backend": "TRITON_ATTN", "env": {}},
    # vLLM's own quantized caches as references; the backend is left to vLLM
    # (it rejects an explicit TURBOQUANT selection with this dtype).
    "fp8": {"backend": None, "kv_cache_dtype": "fp8", "env": {}},
    "tq_k8v4": {"backend": None, "kv_cache_dtype": "turboquant_k8v4", "env": {}},
    "dkv_all": {"backend": "CUSTOM", "env": {"DKV_VLLM_CODEC": "k8v4",
                                             "DKV_VLLM_ROUTE_TOKENS": "0"}},
    "dkv_route": {"backend": "CUSTOM", "env": {"DKV_VLLM_CODEC": "k8v4",
                                               "DKV_VLLM_ROUTE_TOKENS": "2048",
                                               "DKV_VLLM_WINDOW": "1024"}},
}


def needle_prompts(tok, ctx, depths=(0.05, 0.25, 0.45, 0.65, 0.85, 0.95)):
    rnd = random.Random(7)
    out = []
    per = len(tok(FILLER).input_ids)
    n_fill = max(1, (ctx - 120) // per)
    for d in depths:
        key = str(rnd.randint(10000, 99999))
        parts = [FILLER] * n_fill
        parts.insert(int(d * n_fill), f" The pass key is {key}. Remember it. {key} is the pass key. ")
        text = ("There is important info hidden inside a lot of irrelevant text. "
                "Find it and memorize it.\n" + "".join(parts) +
                "\nWhat is the pass key? The pass key is")
        out.append((text, key))
    return out


# CUDA graphs ON for every arm by default (stock included, so the comparison
# is like for like); DKV_GATE_EAGER=1 runs all arms eagerly.
EAGER = os.environ.get("DKV_GATE_EAGER", "0") == "1"


def run_arm(name, model, ctx, out_path):
    spec = ARMS[name]
    os.environ.update(spec["env"])
    if name.startswith("dkv"):
        os.environ["DKV_VLLM_STATS"] = out_path + ".stats.json"
    from vllm import LLM, SamplingParams
    kw = dict(model=model, gpu_memory_utilization=0.80,
              max_model_len=ctx + 512, enforce_eager=EAGER, seed=0, max_num_seqs=8)
    if spec["backend"]:
        kw["attention_backend"] = spec["backend"]
    if spec.get("kv_cache_dtype"):
        kw["kv_cache_dtype"] = spec["kv_cache_dtype"]
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    greedy = SamplingParams(temperature=0.0, max_tokens=48)
    res = {"arm": name}
    res["short"] = [list(o.outputs[0].token_ids) for o in llm.generate(SHORT, greedy)]
    nd = needle_prompts(tok, ctx)
    outs = llm.generate([p for p, _ in nd], SamplingParams(temperature=0.0, max_tokens=12))
    res["needle"] = [(k, o.outputs[0].text) for (_p, k), o in zip(nd, outs)]
    # decode-only rate: (128-token run) - (1-token run) on the same prompt, so
    # the prefill is subtracted out; prefix caching is off by default here.
    times = {}
    for n in (1, 128):
        t0 = time.perf_counter()
        o = llm.generate([nd[2][0]], SamplingParams(temperature=0.0, max_tokens=n,
                                                    ignore_eos=True))
        times[n] = (time.perf_counter() - t0, len(o[0].outputs[0].token_ids))
    res["decode"] = {"tokens": times[128][1] - times[1][1],
                     "seconds": times[128][0] - times[1][0], "ttft_s": times[1][0]}
    json.dump(res, open(out_path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--role", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.role:
        return run_arm(a.role, a.model, a.ctx, a.out)
    res = {}
    for arm in a.arms:
        out = f"/tmp/dkv_vllm_p23_{arm}.json"
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--role", arm, "--model", a.model,
                            "--ctx", str(a.ctx), "--out", out],
                           capture_output=True, text=True)
        log = p.stdout + p.stderr
        if p.returncode:
            errs = [l for l in log.splitlines()
                    if re.search(r"Error|error:|NotImplemented|not supported", l)
                    and "deprecated" not in l]
            print(f"[{arm}] FAILED -- first error lines:\n" + "\n".join(errs[:8]))
            continue
        r = json.load(open(out))
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)
        r["capacity"] = int(m.group(1).replace(",", "")) if m else None
        st = out + ".stats.json"
        r["stats"] = json.load(open(st)) if os.path.exists(st) else None
        r["wall"] = time.perf_counter() - t0
        res[arm] = r
    ref = res.get("stock")
    print(f"\n{'arm':<10} {'short exact':>11} {'prefix':>7} {'needle':>7} "
          f"{'KV tokens':>10} {'x stock':>7} {'dec tok/s':>9} {'pages read':>10}")
    for arm, r in res.items():
        ex, pre = "-", "-"
        if ref and arm != "stock":
            pairs = list(zip(ref["short"], r["short"]))
            ex = f"{sum(x == y for x, y in pairs)}/{len(pairs)}"
            lens = []
            for x, y in pairs:
                n = 0
                while n < min(len(x), len(y)) and x[n] == y[n]:
                    n += 1
                lens.append(n)
            pre = f"{sum(lens) / len(lens):.1f}"
        nd = sum(k in txt for k, txt in r["needle"])
        cap = r["capacity"]
        rel = (f"{cap / ref['capacity']:.2f}" if ref and ref.get("capacity") and cap else "-")
        tps = r["decode"]["tokens"] / r["decode"]["seconds"]
        s = r.get("stats") or {}
        pr = (f"{s['pages_read'] / s['pages_total']:.2f}"
              if s.get("pages_total") else "-")
        print(f"{arm:<10} {ex:>11} {pre:>7} {nd:>3}/{len(r['needle']):<3} "
              f"{cap if cap else '-':>10} {rel:>7} {tps:>9.1f} {pr:>10}")
    json.dump(res, open("/tmp/dkv_vllm_p23_summary.json", "w"))


if __name__ == "__main__":
    main()
