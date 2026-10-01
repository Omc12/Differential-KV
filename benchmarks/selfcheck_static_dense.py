#!/usr/bin/env python3
"""Gate for the preallocated dense baseline: it must compute what dense computes.

A StaticCache changes where the cache lives, not the arithmetic -- except that
it attends over the whole preallocated length under a mask, which selects a
different SDPA kernel and so a different bf16 rounding. Exact text equality is
therefore too strict a test (on Qwen3.5-4B the two greedy continuations agree
for 11 tokens and then flip on a near-tie). This gate teacher-forces BOTH
caches along the same token sequence and compares the next-token
distributions at every step. A real defect -- a layer reading the wrong
state, a position offset -- shows up as large KL on most steps; rounding shows
up as small KL everywhere.

Pass: top-1 agrees on >= 90% of steps and the max per-step KL < 0.02.

    python benchmarks/selfcheck_static_dense.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAX_KL = 0.02
MIN_TOP1 = 0.90


def forced(model, ids, cont, static_len):
    """Prefill ids, then feed cont token by token; return per-step logits."""
    import torch
    import kv_baselines as KB
    past, last = KB.chunked_prefill(model, ids, "cuda", 1024, static_len=static_len)
    out = [last]
    cur = len(ids)
    with torch.no_grad():
        for t in cont:
            o = model(input_ids=torch.tensor([[t]], device="cuda"),
                      position_ids=torch.tensor([[cur]], device="cuda"),
                      past_key_values=past, use_cache=True)
            past = o.past_key_values
            out.append(o.logits[0, -1].float())
            cur += 1
    return torch.stack(out)


def check(model_id: str, ctx: int = 4096, gen: int = 32) -> bool:
    import torch
    import kv_baselines as KB
    from run_longbench_cuda import load_plain
    from context_ladder import build_filler
    tok, model = load_plain(model_id, "nf4", False)
    ids = tok(build_filler(tok, ctx), add_special_tokens=False).input_ids[:ctx]
    ref = KB.run_baseline(model, tok, ids, "dense", "cuda", gen, set(), 1024, {})
    cont = tok(ref["text"], add_special_tokens=False).input_ids[:gen]
    a = forced(model, ids, cont, 0)
    torch.cuda.empty_cache()
    b = forced(model, ids, cont, len(ids) + len(cont) + 1)
    pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    ok = top1 >= MIN_TOP1 and kl.max().item() < MAX_KL
    print(f"  {model_id}: {len(cont) + 1} steps, top-1 agree {top1:.3f}, "
          f"KL mean {kl.mean().item():.2e} max {kl.max().item():.2e} -> "
          f"{'ok' if ok else 'MISMATCH'}", flush=True)
    del model
    torch.cuda.empty_cache()
    return ok


def main():
    from msvc_env import ensure_msvc
    ensure_msvc()
    ok = True
    for m in sys.argv[1:] or ["Qwen/Qwen3.5-4B", "ibm-granite/granite-4.2-8b"]:
        ok &= check(m)
    print("STATIC DENSE SELFCHECK", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
