#!/usr/bin/env python3
"""Why does DKV compress granite 4.1x and gain no ceiling?

The ladder says DKV stores 0.653 GB of KV on granite-4.2-8b at 16k against
dense's 2.684 GB, yet the two arms' peak-memory slopes are identical (0.2230 vs
0.2213 GB per 1k tokens) and both stop at the same rung. Something other than
the compressed store is scaling with context. This attributes it.

Method: after prefill, walk every live CUDA tensor and bucket it by shape.
A full-length KV cache is unmistakable -- 4-D, with one axis equal to the
prompt length -- so if the dense cache is still resident alongside the
compressed pool, this prints it.

Run on both arms and both models; granite is the case in question and
Qwen3.5-4B is the control where the ceiling DOES move.

    python diag_granite_residency.py --model ibm-granite/granite-4.2-8b --ctx 16384
"""
import argparse
import gc
import json
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")


def live_cuda_tensors(ctx_len):
    """Every live CUDA tensor, bucketed. Returns (buckets, total_bytes)."""
    import torch
    seen, buckets, total = set(), defaultdict(lambda: [0, 0]), 0
    for o in gc.get_objects():
        try:
            t = None
            if isinstance(o, torch.Tensor):
                t = o
            elif hasattr(o, "data") and isinstance(getattr(o, "data", None), torch.Tensor):
                t = o.data
            if t is None or not t.is_cuda or t.numel() == 0:
                continue
            key = (t.data_ptr(), tuple(t.shape))
            if key in seen:
                continue
            seen.add(key)
        except Exception:                                          # noqa: BLE001
            continue
        nb = t.numel() * t.element_size()
        total += nb
        shp = tuple(t.shape)
        # A KV cache entry is 4-D with the sequence axis at the prompt length.
        kv_like = (t.dim() == 4 and any(abs(s - ctx_len) <= 64 for s in shp))
        tag = "KV-SHAPED (seq axis == prompt len)" if kv_like else (
            "4-D other" if t.dim() == 4 else
            "3-D" if t.dim() == 3 else
            "2-D (weights/factors)" if t.dim() == 2 else "1-D/other")
        b = buckets["%s  %s" % (tag, str(t.dtype).replace("torch.", ""))]
        b[0] += nb
        b[1] += 1
    return buckets, total


def run_point(a):
    import torch
    out = {"arm": a.arm, "model": a.model, "ctx": a.ctx}
    if a.arm == "dkv":
        os.environ.setdefault("DKV_RSVD_SEED", "1234")
        os.chdir(ACTIVE)
        sys.path.insert(0, ACTIVE)
        from serving.hf_dkv_wrapper import DKVHFWrapper
        w = DKVHFWrapper(model_id=a.model,
                         config={"preset": a.preset, "quantization": a.quant or None})
        w.ensure_loaded()
        tok = w.tokenizer
    else:
        sys.path.insert(0, HERE)
        from run_longbench_cuda import load_plain
        tok, model = load_plain(a.model, a.quant, False)

    sys.path.insert(0, HERE)
    from context_ladder import build_filler
    prompt = build_filler(tok, a.ctx)
    ids = tok(prompt, add_special_tokens=False).input_ids
    out["ctx_actual"] = len(ids)

    torch.cuda.synchronize()
    out["weights_gb"] = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    if a.arm == "dkv":
        w.active_session = "diag"
        w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                   repetition_penalty=1.0)
    else:
        import kv_baselines as KB
        past, _ = KB.chunked_prefill(model, ids, "cuda", a.chunk)
        out["_hold"] = True          # keep `past` alive across the walk
    torch.cuda.synchronize()

    out["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    out["resident_gb"] = torch.cuda.memory_allocated() / 1e9
    buckets, total = live_cuda_tensors(out["ctx_actual"])
    out["walked_gb"] = total / 1e9
    out["buckets"] = {k: {"gb": round(v[0] / 1e9, 4), "n": v[1]}
                      for k, v in sorted(buckets.items(),
                                         key=lambda kv: -kv[1][0])}
    if a.arm != "dkv":
        del past
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--preset", default="mid")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--arms", default="dense,dkv")
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    if a.arm:
        return run_point(a)
    for arm in a.arms.split(","):
        arm = arm.strip()
        print("\n### %s  %s @ %d" % (arm, a.model.split("/")[-1], a.ctx), flush=True)
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--arm", arm,
             "--model", a.model, "--ctx", str(a.ctx), "--quant", a.quant,
             "--preset", a.preset, "--chunk", str(a.chunk)],
            capture_output=True, text=True, cwd=REPO)
        line = ""
        for ln in p.stdout.splitlines():
            if ln.startswith("{"):
                line = ln
        if not line:
            print((p.stdout or "")[-1200:] + (p.stderr or "")[-1200:])
            continue
        r = json.loads(line)
        print("  weights %.2f  peak %.2f  resident-after %.2f  walked %.2f GB"
              % (r["weights_gb"], r["peak_gb"], r["resident_gb"], r["walked_gb"]))
        for k, v in r["buckets"].items():
            if v["gb"] >= 0.01:
                print("     %-46s %7.3f GB  (%d tensors)" % (k, v["gb"], v["n"]))


if __name__ == "__main__":
    main()
