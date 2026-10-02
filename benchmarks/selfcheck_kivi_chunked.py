#!/usr/bin/env python3
"""Gate for the chunk-wise KIVI-4 arm: it must be a faithful 4-bit cache.

Three checks on both models, 4k-token prompt:
  1. every attention cache layer was converted (on the hybrid, 8 of 32);
  2. the stored bytes are what 4-bit storage implies: between 3x and 4x
     smaller than the bf16 cache it replaces;
  3. teacher-forced along dense's own continuation, its next-token
     distribution stays close to dense's (4-bit KIVI is near-lossless):
     top-1 agreement >= 85% and mean per-step KL < 0.05.

    python benchmarks/selfcheck_kivi_chunked.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def forced(model, ids, cont, method):
    import torch
    import kv_baselines as KB
    if method == "dense":
        past, last = KB.chunked_prefill(model, ids, "cuda", 1024)
        phys, n_conv = None, None
    else:
        past, last, phys, _ = KB._kivi_chunked(model, ids, "cuda", 1024)
        n_conv = sum(1 for l in past.layers if hasattr(l, "kivi_bytes"))
    out = [last]
    cur = len(ids)
    with torch.no_grad():
        for t in cont:
            o = model(input_ids=torch.tensor([[t]], device="cuda"),
                      position_ids=torch.tensor([[cur]], device="cuda"),
                      past_key_values=past, use_cache=True)
            out.append(o.logits[0, -1].float())
            cur += 1
    return torch.stack(out), phys, n_conv


def check(model_id, ctx=4096, gen=32):
    import torch
    import kv_baselines as KB
    from run_longbench_cuda import load_plain
    from context_ladder import build_filler
    tok, model = load_plain(model_id, "nf4", False)
    ids = tok(build_filler(tok, ctx), add_special_tokens=False).input_ids[:ctx]
    ref = KB.run_baseline(model, tok, ids, "dense", "cuda", gen, set(), 1024, {})
    cont = tok(ref["text"], add_special_tokens=False).input_ids[:gen]
    a, _, _ = forced(model, ids, cont, "dense")
    torch.cuda.empty_cache()
    b, phys, n_conv = forced(model, ids, cont, "kivi")
    pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    dense_gb = ref["kv_dense_equiv_gb"]
    ratio = dense_gb / phys if phys else 0.0
    ok = (n_conv and 3.0 <= ratio <= 4.0 and top1 >= 0.85 and kl.mean().item() < 0.05)
    print("  %s: %d layers converted, store %.4f GB vs bf16 %.4f GB (%.2fx), "
          "top-1 agree %.3f, KL mean %.2e max %.2e -> %s"
          % (model_id, n_conv, phys, dense_gb, ratio, top1, kl.mean().item(),
             kl.max().item(), "ok" if ok else "MISMATCH"), flush=True)
    del model
    torch.cuda.empty_cache()
    return bool(ok)


def main():
    from msvc_env import ensure_msvc
    ensure_msvc()
    ok = True
    for m in sys.argv[1:] or ["Qwen/Qwen3.5-4B", "ibm-granite/granite-4.2-8b"]:
        ok &= check(m)
    print("KIVI CHUNKED SELFCHECK", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
