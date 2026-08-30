#!/usr/bin/env python3
"""Would a relevance THRESHOLD pick fewer or more blocks than a fixed K?

WHY THIS EXISTS
---------------
Routing currently takes a fixed top-K. The obvious improvement is to take blocks
until they cover some share of the relevance mass -- top-p over blocks -- so an
easy lookup costs 1-2 blocks and a synthesis query gets as many as it needs.
Whether that is CHEAPER or more expensive than fixed K is an empirical question
about the score distribution, and this measures it before anything is built.

READ THE CAVEAT IN THE OUTPUT. The router's score is a MAX over each block's
anchor + top-R residual keys, i.e. an upper bound on that block's best attention
logit -- NOT the mass the block would actually receive. Softmaxing it (what this
probe does, and what a threshold rule would do) therefore over-weights SPIKY
blocks: one outlier key beats a block holding many moderately relevant tokens.
That bias favours needle retrieval and works against synthesis, which is already
the metric a small K hurts. If the numbers here argue for thresholding, the score
probably has to become mass-like (logsumexp over the block's keys) at the same
time -- the selection rule alone is only half of it.

NON-INVASIVE. MLX: wraps _block_relevance_residual at module level, exactly as
probe_mlx_router_rank.py does. CUDA: installs query_router._ROUTE_DIST_HOOK,
which is None in production and costs one identity check per route.

    ENGINE=mlx  SEEDS=1,2,3 CTX=16000 python colab/probe_relevance_dist.py
    ENGINE=cuda SEEDS=1,2,3 CTX=16000 python colab/probe_relevance_dist.py
"""
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)

from colab.linkbench_cuda import build                            # noqa: E402

ENGINE = os.environ.get("ENGINE", "mlx").lower()
THRESHOLDS = [0.5, 0.75, 0.9, 0.95, 0.99]
samples = []          # (nb, top1_share, {threshold: K_needed})


def _record(scores_desc_cumsum, nb, top1):
    ks = {}
    for t in THRESHOLDS:
        ks[t] = next((i + 1 for i, c in enumerate(scores_desc_cumsum) if c >= t), nb)
    samples.append((nb, top1, ks))


def _run_mlx(seeds, model):
    import mlx.core as mx
    import serving.mlx_dkv_wrapper as W
    orig = W._block_relevance_residual

    def traced(*a, **kw):
        rel = orig(*a, **kw)
        try:
            r = rel.astype(mx.float32)
            if r.ndim == 1 and r.shape[0] > 1:
                p = mx.softmax(r, axis=-1)
                ps = mx.sort(p)[::-1]
                _record(mx.cumsum(ps).tolist(), int(r.shape[0]), float(ps[0].item()))
        except Exception:                                         # noqa: BLE001
            pass
        return rel

    W._block_relevance_residual = traced
    try:
        from serving.mlx_dkv_wrapper import MLXDKVWrapper
        w = MLXDKVWrapper(model_id=model, config={"preset": "mid"})
        w.ensure_loaded()
        print(f"  [mlx] block_size={w.manager.block_size} K={w.manager.topk_blocks}",
              flush=True)
        _drive(w, seeds)
    finally:
        W._block_relevance_residual = orig


def _run_cuda(seeds, model):
    import torch
    from native_core.srl import query_router as QR

    def hook(relevance, layer_idx):
        r = relevance.detach().float()
        if r.dim() != 1 or r.numel() < 2:
            return
        p = torch.softmax(r, dim=-1)
        ps, _ = torch.sort(p, descending=True)
        _record(torch.cumsum(ps, 0).tolist(), int(r.numel()), float(ps[0].item()))

    QR._ROUTE_DIST_HOOK = hook
    try:
        # PyTorchDKVHFWrapper, not HFDKVWrapper -- the latter name does not
        # exist, so this path had never been executed. device= is required too;
        # the wrapper does not infer CUDA.
        from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
        w = PyTorchDKVHFWrapper(model_id=model, config={"preset": "mid"},
                                device="cuda")
        w.ensure_loaded()
        pool = getattr(w.manager, "native_pool", None)
        print(f"  [cuda] micro_block_size={w.micro_block_size} "
              f"observed_span={getattr(pool, 'observed_block_span', '?')} "
              f"routing_topk_default={getattr(pool, 'routing_topk_default', '?')}",
              flush=True)
        _drive(w, seeds)
        print(f"  [cuda] observed_span AFTER the run="
              f"{getattr(pool, 'observed_block_span', '?')}", flush=True)
    finally:
        QR._ROUTE_DIST_HOOK = None


def _drive(w, seeds):
    for seed in seeds:
        body, _q, _ans, _c = build(w.tokenizer, seed)
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False,
            add_generation_prompt=True)
        w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                   top_p=1.0, repetition_penalty=1.0)
        print(f"  seed={seed} done, {len(samples)} router calls so far", flush=True)


def main():
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    seeds = [int(x) for x in os.environ.get("SEEDS", "1,2,3").split(",") if x.strip()]
    model = os.environ.get("MODEL", "mlx-community/Qwen3.5-2B-4bit"
                           if ENGINE == "mlx" else "Qwen/Qwen3.5-2B")
    (_run_mlx if ENGINE == "mlx" else _run_cuda)(seeds, model)

    if not samples:
        print("\nNO ROUTER CALLS OBSERVED — the router never ran on these prompts.\n"
              "That is itself the result: check the block count against K "
              "(routing is skipped when nb <= k_eff).")
        return

    nbs = Counter(s[0] for s in samples)
    top1 = sorted(s[1] for s in samples)
    print(f"\n{len(samples)} router calls   blocks/call: "
          + ", ".join(f"nb={n}:{c}" for n, c in nbs.most_common(3)))
    print(f"top-1 block's share of softmaxed relevance: "
          f"median {top1[len(top1)//2]:.3f}  min {top1[0]:.3f}  max {top1[-1]:.3f}")
    print(f"\n{'threshold':>10} {'median K':>9} {'p90 K':>7} {'max K':>7}   most common K")
    for t in THRESHOLDS:
        ks = sorted(s[2][t] for s in samples)
        dist = Counter(ks)
        top = "  ".join(f"K={k}:{100 * c / len(ks):.0f}%" for k, c in dist.most_common(4))
        print(f"{t:>10.2f} {ks[len(ks)//2]:>9d} {ks[int(0.9*(len(ks)-1))]:>7d} "
              f"{ks[-1]:>7d}   {top}")
    print("\nCompare the median/p90 columns against the fixed K printed above.")
    print("p90 is the one that decides cost: a rule that is usually cheap and")
    print("occasionally takes everything does not bound the decode cache.")
    print("\nCAVEAT (see module docstring): these are softmaxed MAX-per-block")
    print("scores, so the distribution is biased toward spiky blocks.")


if __name__ == "__main__":
    main()
