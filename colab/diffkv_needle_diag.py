#!/usr/bin/env python3
"""A100 diagnostic for the deep-needle fix. Runs the EXACT test_niah.py prompt at
8000/0.1 on CUDA and instruments the needle block's fate:
  - is it in the decode COMPRESSED list?  (metadata-sync fix, Part 1)
  - how many EXACT residuals does its pool slot hold?  (force_exact fix, Part 2 — GPU path)
  - is it ROUTED (selected) at decode?  (routing / TOPK)
and prints the generated text.

Run (transformers MUST be 4.46.3):
  DIFFKV_TOPK_BLOCKS=64 python colab/diffkv_needle_diag.py --model Qwen/Qwen2.5-0.5B-Instruct
Also try DIFFKV_TOPK_BLOCKS=0 (attend-all) to isolate routing from fidelity.
"""
import os, sys, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    ft = tokenizer.encode(filler, add_special_tokens=False)
    nt = tokenizer.encode(needle + "\n", add_special_tokens=False)
    tgt = max(100, context_length - len(nt) - 100)
    allf = (ft * ((tgt // len(ft)) + 1))[:tgt]
    idx = int(len(allf) * depth)
    p1 = tokenizer.decode(allf[:idx]); p2 = tokenizer.decode(allf[idx:])
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + p1 + "\n" + needle + "\n" + p2 + "\n\n"
            + question + "<|im_end|>\n<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--depth", type=float, default=0.1)
    args = ap.parse_args()

    from native_core.kv_runtime_manager import KVRuntimeManager
    _orig = KVRuntimeManager.get_cached_decode_blocks
    seen = {"n": 0}

    def _patched(self, session_id, layer_idx, device):
        res = _orig(self, session_id, layer_idx, device)
        if layer_idx == 0 and seen["n"] < 1:
            block_indices_t, dense_blocks, anchor_indices_gpu, *_ = res
            comp = anchor_indices_gpu.tolist() if anchor_indices_gpu is not None else []
            mgr = getattr(self, "_streaming_mgr", None)
            pool = getattr(self, "native_pool", None)
            # find the needle block (its anchor is near depth*context; search skip blocks with a digit)
            needle_anchor = None; needle_pidx = None
            if mgr is not None:
                for b in mgr.session_blocks.get(session_id, {}).get(layer_idx, []):
                    if getattr(b, "skip_compression", False) and getattr(b, "pool_idx", None) is not None:
                        needle_anchor = int(getattr(b, "anchor_idx", -1)); needle_pidx = b.pool_idx
                        break
            nres = 0
            if pool is not None and needle_pidx is not None:
                rp = pool.residual_K_positions[needle_pidx]
                nres = int((rp >= 0).sum().item())
            in_comp = needle_anchor in comp if needle_anchor is not None else "?"
            print(f"[DIAG L0] TOPK={os.environ.get('DIFFKV_TOPK_BLOCKS','default16')} "
                  f"n_compressed_routed={len(comp)} | needle_anchor={needle_anchor} pool_idx={needle_pidx} "
                  f"needle_in_routed={in_comp} needle_n_residuals={nres}", flush=True)
            seen["n"] += 1
        return res
    KVRuntimeManager.get_cached_decode_blocks = _patched

    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = DiffKVHFWrapper(args.model, config={"rank": 32}, device=device)
    prompt = make_niah_prompt(w.tokenizer, args.ctx, args.depth, needle, question)
    ptoks = len(w.tokenizer.encode(prompt))
    w.generate(prompt=prompt, max_new_tokens=12, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
    sid = w.active_session or "default"
    gen = w._session_token_ids.get(sid, [])[ptoks:]
    gt = w.tokenizer.decode(gen, skip_special_tokens=True)
    print(f"[DIAG] ctx={args.ctx} depth={args.depth} → {'FOUND ✓' if '847291' in gt else 'MISS ✗'}  gen={gt!r}", flush=True)


if __name__ == "__main__":
    main()
