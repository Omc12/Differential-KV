#!/usr/bin/env python3
"""What RANK does MLX's router give the needle's block?

WHY
---
The CUDA route trace (DKV_ROUTE_TRACE=1) established that CUDA's router DROPS
the needle's block: at 8k@depth0.5 it sits at rank 16-37 of 38 with k=16 and is
never selected, and every 8k/32k case that passes is one where the needle is
reachable WITHOUT the router (the sink block, or inside the dense recency
window). So the failure is routing, not the kernel.

MLX passes all nine cases. MLX also gathers residuals ONLY for the blocks its
router selected --

    rk = mx.take(comp_res_k, sel, 0)          (mlx_dkv_wrapper.py:1002, :4033)

-- so if MLX's router dropped the needle's block, MLX would fail too. It does
not. Therefore MLX's router KEEPS a block CUDA's router ranks near last, on the
same prompt, with the same k_eff (both resolve to 16: MLX block_size=256 ->
max(16, 4096//256), CUDA observed_block_span=256 -> max(16, 4096//257)).

This prints MLX's rank for that block so the two can be compared directly.

MLX IS NOT MODIFIED. _block_relevance_residual is wrapped at the MODULE level
from here; mlx_dkv_wrapper.py is untouched and the original function still does
all the arithmetic. The wrapper only observes its return value.

    python colab/probe_mlx_router_rank.py              # 32k@depth0.9
    python colab/probe_mlx_router_rank.py --depth 0.5  # the 8k control
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

NEEDLE = "ZEBRA-4471-QUARTZ"

# verbatim from validate_cuda_dkv.py / mlx_needle_parity.py
FILLER = [
    "The morning fog rolled over the hills before the sun broke through the clouds.",
    "Researchers published a new dataset covering climate trends across five continents.",
    "The old library smelled of dust and aging paper, a comfort to regular visitors.",
    "Markets fluctuated throughout the week as investors weighed new economic data.",
    "A gentle breeze carried the scent of pine through the quiet mountain trail.",
    "The committee reviewed dozens of proposals before selecting a final design.",
    "Local farmers reported a strong harvest season despite the unpredictable weather.",
    "The orchestra rehearsed late into the evening, perfecting the final movement.",
]


def build(n_filler, depth):
    filler = [random.choice(FILLER) for _ in range(n_filler)]
    at = int(len(filler) * depth)
    needle = (f"Remember this important code: {NEEDLE}. "
              "This is the only code you need to remember.")
    parts = filler[:at] + [needle] + filler[at:]
    parts.append("Question: What was the important code mentioned in this "
                 "text? Reply with only the code.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--label", default="32k", choices=["2k", "8k", "32k"])
    ap.add_argument("--depth", type=float, default=0.9)
    ap.add_argument("--max-lines", type=int, default=40)
    args = ap.parse_args()

    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)

    import mlx.core as mx
    from serving import mlx_dkv_wrapper as W

    # The validator seeds ONCE and builds all nine cases in sequence, so the
    # filler for a given case depends on every draw before it. Replay the whole
    # sequence or this probes a DIFFERENT PROMPT than the one that fails.
    cases = [("2k", 200, 0.0), ("2k", 200, 0.5), ("2k", 200, 0.9),
             ("8k", 800, 0.0), ("8k", 800, 0.5), ("8k", 800, 0.9),
             ("32k", 2400, 0.0), ("32k", 2400, 0.5), ("32k", 2400, 0.9)]
    random.seed(5)
    ctx = None
    for label, n_filler, depth in cases:
        c = build(n_filler, depth)
        if label == args.label and abs(depth - args.depth) < 1e-9:
            ctx = c
    if ctx is None:
        print(f"no case matches {args.label}@{args.depth}")
        return

    w = W.MLXDKVWrapper(model_id=args.model, config={"preset": "mid"})
    w.ensure_loaded()

    prompt = w.tokenizer.apply_chat_template(
        [{"role": "user", "content": ctx}], tokenize=False,
        add_generation_prompt=True)
    ntok = len(w.tokenizer.encode(prompt))
    nchar = prompt.find(NEEDLE)
    needle_tok = len(w.tokenizer.encode(prompt[:nchar]))
    bs = w.manager.block_size
    target_block = needle_tok // bs
    print(f"[probe] {args.label}@depth{args.depth} — {ntok} tok, needle at "
          f"token ~{needle_tok}, block_size={bs} -> block index {target_block}")
    print(f"[probe] topk_blocks={w.manager.topk_blocks} "
          f"topk_frac={w.manager.topk_frac} "
          f"route_residuals={w.manager.route_residuals} "
          f"max_residual={w.manager.max_residual} "
          f"decode_cache={w.manager._decode_cache}")

    # Observe the RETURN of the real function. No modification to MLX: the
    # original still computes everything; this only reads the result.
    orig = W._block_relevance_residual
    state = {"n": 0}

    def traced(q, comp_anc_k, comp_res_k, res_valid, scale, gpk):
        rel = orig(q, comp_anc_k, comp_res_k, res_valid, scale, gpk)
        if state["n"] < args.max_lines:
            try:
                r = mx.array(rel)
                nb = int(r.shape[0])
                if target_block < nb:
                    k_eff = w.manager.topk_blocks
                    if w.manager.topk_blocks > 0 and w.manager.topk_frac > 0.0:
                        k_eff = max(k_eff, int(nb * w.manager.topk_frac))
                    order = mx.argsort(-r)
                    ol = [int(x) for x in order.tolist()]
                    rank = ol.index(target_block)
                    cut = float(r[ol[min(k_eff, nb) - 1]].item())
                    state["n"] += 1
                    print(f"[MLX ROUTE] block={target_block} rank={rank}/{nb} "
                          f"k={k_eff} kept={rank < k_eff} "
                          f"rel={float(r[target_block].item()):.5f} "
                          f"top={float(r[ol[0]].item()):.5f} cut={cut:.5f}",
                          flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"[MLX ROUTE] trace failed: {e}", flush=True)
        return rel

    W._block_relevance_residual = traced
    try:
        out = w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                         top_p=1.0, repetition_penalty=1.0)
    finally:
        W._block_relevance_residual = orig

    ans = out.rsplit("assistant", 1)[-1].strip()
    hit = "".join(c for c in NEEDLE.upper() if c.isalnum()) in \
          "".join(c for c in ans.upper() if c.isalnum())
    print(f"\n[probe] MLX answer: {ans[:60]!r}  recall={'HIT' if hit else 'MISS'}")
    if state["n"] == 0:
        print("[probe] the router never ran -- nb <= k_eff, or the decode path "
              "did not reach _block_relevance_residual. That is itself the "
              "answer: MLX is not routing at this length.")


if __name__ == "__main__":
    main()
