#!/usr/bin/env python3
"""Where does decode time go, per DKV function? (diagnostic)

Generates from one long prompt with the shipped decode path and wraps the
decode-side functions with synchronized timers, so each one's share of the
per-token time is measured directly rather than inferred from kernel names.
Run once per configuration (environment), e.g.:

    DKV_STREAMING_COMPRESS=1 python benchmarks/profile_hybrid_decode.py
    DKV_STREAMING_COMPRESS=1 DKV_KEY_QUANT=pc4 DKV_RESID_ATTN=1 \\
        python benchmarks/profile_hybrid_decode.py
"""
import argparse
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--ruler", default="16384/niah_multikey_2:0-0")
    ap.add_argument("--gen", type=int, default=48)
    args = ap.parse_args()

    import torch
    os.environ["DKV_DISABLE_CUDA_GRAPH"] = "1"
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    from msvc_env import ensure_msvc
    ensure_msvc()
    from serving.decode_config import apply_best_decode_defaults
    apply_best_decode_defaults()
    from serving.hf_dkv_wrapper import DKVHFWrapper
    import runtime.dkv_attention as DA
    import native_core.sparse_decode.remat_cache as RC
    import native_core.sparse_decode.triton_fused_decode as TF
    import runtime.native_block_pool as NBP
    from decode_fidelity import build_items

    acc = defaultdict(float)
    cnt = defaultdict(int)
    state = {"decode": False}

    def timed(mod, name, label=None):
        fn = getattr(mod, name)
        label = label or name

        def w(*a, **k):
            if not state["decode"]:
                return fn(*a, **k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            acc[label] += time.perf_counter() - t0
            cnt[label] += 1
            return r
        setattr(mod, name, w)

    timed(DA, "_remat_attend_impl", "remat_attend (total)")
    timed(TF, "_gather_routed_blocks_for_kernel", "gather")
    timed(RC, "reconstruct_blocks", "reconstruct")
    timed(RC, "attend_with_remat", "attend_with_remat (SDPA)")
    timed(NBP.NativeBlockPool, "get_key_deltas", "key dequant")
    timed(NBP.NativeBlockPool, "_residual_k_from_codes", "K residual from codes")

    w = DKVHFWrapper(model_id=args.model, config={"preset": "mid", "quantization": "nf4"})
    w.ensure_loaded()
    tok, model = w.tokenizer, w.model
    (key, prompt), = build_items(tok, args.model, [], [args.ruler], 12000)

    # mark decode steps: model forwards with a single new token
    def pre(_m, a, kw):
        ii = kw.get("input_ids") if kw.get("input_ids") is not None else (a[0] if a else None)
        state["decode"] = ii is not None and ii.shape[1] == 1
        if state["decode"]:
            torch.cuda.synchronize()
            state["t0"] = time.perf_counter()

    def post(_m, _a, _o):
        if state["decode"]:
            torch.cuda.synchronize()
            acc["TOKEN (whole step)"] += time.perf_counter() - state["t0"]
            cnt["TOKEN (whole step)"] += 1
        state["decode"] = False

    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.register_forward_hook(post)
    w.active_session = "prof"
    w.generate(prompt, max_new_tokens=args.gen, temperature=0.0, top_p=1.0,
               repetition_penalty=1.0)
    n_tok = max(1, cnt["TOKEN (whole step)"])
    try:
        rc = w.manager.decode_workspace.get("prof", {}).get("_remat_cache")
        free, total = torch.cuda.mem_get_info()
        print("[remat cache] hits=%s misses=%s entries=%s  free=%.2f GB of %.2f" % (
            getattr(rc, "hits", None), getattr(rc, "misses", None),
            len(getattr(rc, "_store", {}) or {}), free / 1e9, total / 1e9))
    except Exception as e:                                       # noqa: BLE001
        print("[remat cache] unavailable:", e)
    env = {k: os.environ[k] for k in ("DKV_STREAMING_COMPRESS", "DKV_KEY_QUANT",
                                      "DKV_RESID_ATTN", "DKV_REMAT_CACHE") if k in os.environ}
    print("\n=== %s  %s  (%d decode steps)" % (key, env, n_tok))
    tot = acc["TOKEN (whole step)"] / n_tok * 1e3
    for k in sorted(acc, key=lambda k: -acc[k]):
        ms = acc[k] / n_tok * 1e3
        print("  %-28s %8.2f ms/token  %5.1f%%  (%d calls)" % (k, ms, 100 * ms / tot, cnt[k]))


if __name__ == "__main__":
    main()
