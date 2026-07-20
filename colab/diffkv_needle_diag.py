#!/usr/bin/env python3
"""A100 diagnostic for the deep-needle fix. Runs the EXACT test_niah.py prompt at
8000/0.1 on CUDA and instruments:
  - the router (route_blocks_relevance): the K it reads, N candidate blocks, and
    whether the needle's anchor is in the SELECTED set;
  - the needle block's pool residual count.
Flags:
  --no-hook   : do NOT monkeypatch get_cached_decode_blocks (to test if that hook
                is what makes the diagnostic pass vs the plain pytest).
  --max-new N : generated tokens (default 16, matches test_niah.py).

Run (transformers MUST be 4.46.3):
  DIFFKV_TOPK_BLOCKS=64 python colab/diffkv_needle_diag.py --model Qwen/Qwen2.5-0.5B-Instruct
  DIFFKV_TOPK_BLOCKS=64 python colab/diffkv_needle_diag.py --no-hook
"""
import os, sys, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Mirror the PRODUCTION decode config by default: SPARSE_BIAS=auto is the serving
# default (decode_config.py) and forks to the non-combined merge path. Running
# without it silently exercised the combined kernel — a path production never uses
# — which masked the deep-needle bug. Set DIFFKV_SPARSE_BIAS=0 to A/B the combined
# kernel. TOPK is left unset so the block_size-derived pool default (=64) applies.
os.environ.setdefault("DIFFKV_SPARSE_BIAS", "auto")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))
import torch


def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    target_filler_tokens = context_length - len(needle_tokens) - 100
    if target_filler_tokens < 0:
        target_filler_tokens = 100
    num_repeats = (target_filler_tokens // len(filler_tokens)) + 1
    all_filler_tokens = (filler_tokens * num_repeats)[:target_filler_tokens]
    insert_idx = int(len(all_filler_tokens) * depth)
    part1_text = tokenizer.decode(all_filler_tokens[:insert_idx])
    part2_text = tokenizer.decode(all_filler_tokens[insert_idx:])
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1_text + "\n" + needle + "\n" + part2_text + "\n\n"
        + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--depth", type=float, default=0.1)
    ap.add_argument("--no-hook", action="store_true")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    NEEDLE_ANCHOR = {"v": None}

    # ── Router instrument (device-agnostic): log K + needle selection ──
    import native_core.srl.query_router as qr
    _orig_rbr = qr.route_blocks_relevance
    seen_r = {"n": 0}
    def _rbr(Q, pool, block_indices, anchor_indices, scale, cos=None, sin=None, srl_state=None):
        sel = _orig_rbr(Q, pool, block_indices, anchor_indices, scale, cos=cos, sin=sin, srl_state=srl_state)
        if seen_r["n"] < 1 and NEEDLE_ANCHOR["v"] is not None:
            try:
                topk = int(os.environ.get("DIFFKV_TOPK_BLOCKS", "16"))
                anc = anchor_indices.tolist() if anchor_indices is not None else []
                sel_anc = []
                if sel is not None and sel.numel() > 0:
                    m = (block_indices.unsqueeze(1) == sel.unsqueeze(0)).any(dim=1)
                    sel_anc = anchor_indices[m].tolist()
                na = NEEDLE_ANCHOR["v"]
                print(f"[ROUTER] K_env={topk} N_candidates={len(anc)} n_selected={len(sel_anc)} "
                      f"needle_anchor={na} needle_in_candidates={na in anc} needle_in_SELECTED={na in sel_anc}", flush=True)
                seen_r["n"] += 1
            except Exception as e:
                print(f"[ROUTER] instrument err: {e}", flush=True)
        return sel
    qr.route_blocks_relevance = _rbr

    from native_core.kv_runtime_manager import KVRuntimeManager
    if not args.no_hook:
        _orig = KVRuntimeManager.get_cached_decode_blocks
        seen = {"n": 0}
        def _patched(self, session_id, layer_idx, device):
            res = _orig(self, session_id, layer_idx, device)
            if layer_idx == 0 and seen["n"] < 1:
                _, _, anchor_indices_gpu, *_ = res
                comp = anchor_indices_gpu.tolist() if anchor_indices_gpu is not None else []
                mgr = getattr(self, "_streaming_mgr", None); pool = getattr(self, "native_pool", None)
                na = npidx = None
                if mgr is not None:
                    for b in mgr.session_blocks.get(session_id, {}).get(layer_idx, []):
                        if getattr(b, "skip_compression", False) and getattr(b, "pool_idx", None) is not None:
                            na = int(getattr(b, "anchor_idx", -1)); npidx = b.pool_idx; break
                nres = int((pool.residual_K_positions[npidx] >= 0).sum().item()) if (pool is not None and npidx is not None) else 0
                print(f"[DIAG L0] needle_anchor={na} pool_idx={npidx} needle_in_routed={na in comp} needle_n_residuals={nres}", flush=True)
                seen["n"] += 1
            return res
        KVRuntimeManager.get_cached_decode_blocks = _patched

    # Find the needle anchor from the streaming manager after prefill.
    _orig_gsl = KVRuntimeManager.get_session_sequence_length
    def _find_needle(self, *a, **k):
        if NEEDLE_ANCHOR["v"] is None:
            mgr = getattr(self, "_streaming_mgr", None)
            if mgr is not None:
                for b in mgr.session_blocks.get(a[0] if a else "default", {}).get(0, []):
                    if getattr(b, "skip_compression", False) and getattr(b, "pool_idx", None) is not None:
                        NEEDLE_ANCHOR["v"] = int(getattr(b, "anchor_idx", -1)); break
        return _orig_gsl(self, *a, **k)
    KVRuntimeManager.get_session_sequence_length = _find_needle

    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = DiffKVHFWrapper(args.model, config={"rank": 32}, device=device)
    prompt = make_niah_prompt(w.tokenizer, args.ctx, args.depth, needle, question)
    ptoks = len(w.tokenizer.encode(prompt))
    print(f"[DIAG] no_hook={args.no_hook} max_new={args.max_new} plen={ptoks} "
          f"TOPK={os.environ.get('DIFFKV_TOPK_BLOCKS','default16')}", flush=True)
    w.generate(prompt=prompt, max_new_tokens=args.max_new, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
    sid = w.active_session or "default"
    gen = w._session_token_ids.get(sid, [])[ptoks:]
    gt = w.tokenizer.decode(gen, skip_special_tokens=True)
    print(f"[DIAG] ctx={args.ctx} depth={args.depth} → {'FOUND ✓' if '847291' in gt else 'MISS ✗'}  gen={gt!r}", flush=True)


if __name__ == "__main__":
    main()
