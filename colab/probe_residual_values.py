"""Are the needle's stored residual VALUES correct, or merely SELECTED?

probe_needle_block.py proved the needle is chosen into its block's exact-residual
set (K=12/12) even on the prompt that deterministically answers 'None'. That is
membership, NOT correctness. If the stored values sit in the wrong RoPE frame, or
carry the wrong scale, or were captured pre-RoPE while the pool is post-RoPE, then
membership is meaningless and every downstream suspect is chasing good indices
pointing at bad numbers.

This reconstructs what the DECODER will actually read for the needle's slots --

    K_exact = anchors_K[block]  +  residual_K_values[block][slot]

(the EXACT/substitution form: residuals are anchor-relative true values, so
anchor + residual IS the key, no low-rank term involved) -- and compares it
against ground truth computed from the model's own weights:

    K_true  = RoPE( k_proj(hidden_states[pos]), pos )

Ground truth is captured with a forward pre-hook on each attention module during
DKV's own prefill, so it is the same hidden states DKV itself compressed -- no
second forward, no slice-context error.

    python colab/probe_residual_values.py             # 32k@0.9 (the failing case)
    python colab/probe_residual_values.py --depth 0.5 # a PASSING control

Read it as:
    cos ~ 1.0, rel err small  -> values are right; the loss is downstream of the
        pool entirely (kernel read, merge, or the model genuinely not using it).
    cos far from 1.0          -> the stored residual is WRONG. Compare the same
        number on a passing depth: if that one is also wrong, the residual path is
        broken everywhere and recall survives on redundancy; if only the failing
        depth is wrong, it is position/frame dependent.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

from probe_needle_block import (          # noqa: E402  (path set above)
    NEEDLE, build, needle_token_span, _pool_of,
)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.9)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    a = ap.parse_args()

    try:
        from ACTIVE_RUNTIME.serving import decode_config
        for k, v in getattr(decode_config, "BEST_DECODE_DEFAULTS", {}).items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass
    os.environ.setdefault("DKV_SYNC_COMPRESS", "1")
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")

    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    w = PyTorchDKVHFWrapper(model_id=a.model, config={"mode": "fp16"}, device="cuda")
    w.ensure_loaded()

    # validator's RNG sequence -> the exact prompt that fails
    random.seed(5)
    cases = [("2k", 200, d) for d in (0.0, 0.5, 0.9)]
    cases += [("8k", 800, d) for d in (0.0, 0.5, 0.9)]
    cases += [("32k", 2400, d) for d in (0.0, 0.5, 0.9)]
    ctx = None
    for lbl, n, d in cases:
        c = build(n, d)
        if lbl == "32k" and abs(d - a.depth) < 1e-9:
            ctx = c
    assert ctx is not None, f"no 32k case at depth {a.depth}"

    tok = w.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": ctx}],
                                     tokenize=False, add_generation_prompt=True)
    n_lo, n_hi, n_tok = needle_token_span(tok, prompt)
    print(f"\n=== 32k@depth{a.depth} — {n_tok} tokens, needle at {n_lo}..{n_hi} ===")

    model = w.model if hasattr(w, "model") else w._model
    layers = model.model.layers
    attn_layers = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]

    # ── capture hidden states feeding each attention layer, for the needle only ──
    cap = {i: {} for i in attn_layers}       # layer -> {abs_pos: hidden vector}
    seen = {i: 0 for i in attn_layers}       # running absolute position per layer

    # Capture the needle's WHOLE BLOCK, not just its own tokens. The needle's key
    # is exact (it is a residual), so it can only lose the softmax if OTHER slots
    # score spuriously high -- and those are the non-residual slots rebuilt from
    # anchor + U@V. Measuring them needs ground truth across the block.
    # Anchors sit at multiples of (block_size + 1) = 257; verified against the
    # observed anchors 0/257/.../29041, and 29041 == 113*257.
    _SPAN = 257
    blk_lo = (n_lo // _SPAN) * _SPAN
    blk_hi = blk_lo + _SPAN
    cap_lo, cap_hi = min(n_lo, blk_lo), max(n_hi, blk_hi)

    def mk_hook(li):
        def hook(mod, args, kwargs):
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            if hs is None or hs.dim() != 3:
                return
            start = seen[li]
            T = hs.shape[1]
            seen[li] = start + T
            lo, hi = max(cap_lo, start), min(cap_hi, start + T - 1)
            if lo > hi:
                return
            for p in range(lo, hi + 1):
                cap[li][p] = hs[0, p - start].detach().float().cpu()
        return hook

    handles = [layers[i].self_attn.register_forward_pre_hook(mk_hook(i),
                                                             with_kwargs=True)
               for i in attn_layers]
    try:
        out = w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                         top_p=1.0, repetition_penalty=1.0)
    finally:
        for h in handles:
            h.remove()
    ans = out.rsplit("assistant", 1)[-1].strip().replace("\n", " ")
    got = "".join(c for c in ans.upper() if c.isalnum())
    want = "".join(c for c in NEEDLE.upper() if c.isalnum())
    print(f"model answered {ans[:60]!r} -> "
          f"{'RECALLED' if want in got else 'MISSED'}\n")

    manager, pool = w.manager, _pool_of(w.manager)
    rot = model.model.rotary_emb
    from ACTIVE_RUNTIME.native_core.sparse_decode.triton_fused_decode import (
        _partial_rope_apply,
    )
    try:
        from ACTIVE_RUNTIME.native_core.sparse_decode.triton_fused_decode import (
            pool_stores_rotated_k,
        )
        print(f"pool_stores_rotated_k() = {pool_stores_rotated_k()}  "
              f"(True -> compare against cos_ROT; False -> against cos_RAW)")
    except Exception:
        pass
    print("  RESIDUAL slots (needle, exact form) | LOW-RANK slots (the rest of the block)")
    print(f"{'layer':>5} {'slots':>7} {'cos_ROT':>12} {'cos_RAW':>12} "
          f"{'rel_err':>9} {'lr_cos':>9} {'lr_rel':>9} {'n_lr':>6}")

    for li in attn_layers:
        blocks = manager.get_streaming_blocks("default", li)
        hit = None
        for blk in blocks or []:
            anc = getattr(blk, "anchor_idx", None)
            pidx = getattr(blk, "pool_idx", None)
            if anc is None or pidx is None or pidx < 0:
                continue
            slen = int(pool.seq_lens[pidx].item())
            if anc <= n_lo <= anc + slen:
                hit = (anc, pidx, slen)
                break
        if hit is None:
            print(f"{li:>5}  block not found")
            continue
        anc, pidx, slen = hit

        rk_pos = pool.residual_K_positions[pidx].tolist()
        # .cpu() here, not at the comparison: K_true is built on CPU and mixing
        # devices in the cosine is what the first run tripped over.
        rk_val = pool.residual_K_values[pidx].float().cpu()    # [MAX_RES, H_kv, D]
        anchors_K = pool.anchors_KV[pidx, 0].float().cpu()     # [H_kv, D]

        attn = layers[li].self_attn
        Dh = anchors_K.shape[-1]

        def _gt(p):
            """True post-RoPE key at absolute position p, from the model's weights."""
            if p not in cap[li]:
                return None, None
            h = cap[li][p].to(model.device, dtype=next(attn.parameters()).dtype)
            kt = attn.k_proj(h.unsqueeze(0))
            if getattr(attn, "k_norm", None) is not None:
                kt = attn.k_norm(kt.view(1, -1, Dh))
            kt = kt.view(1, 1, -1, Dh).transpose(1, 2)          # [1, H_kv, 1, D]
            c, s = rot(kt, torch.tensor([[p]], device=model.device))
            k_rot = _partial_rope_apply(kt.float(), c.float().unsqueeze(1),
                                        s.float().unsqueeze(1))
            return k_rot.reshape(-1).cpu(), kt.float().reshape(-1).cpu()

        cos_l, sin_l, np_l, nt_l, raw_l = [], [], [], [], []
        n_ok, n_used, n_fail = 0, 0, 0
        for p in range(n_lo, n_hi + 1):
            off = p - anc - 1
            if off < 0 or off >= slen or p not in cap[li]:
                continue
            try:
                ri = rk_pos.index(off)
            except ValueError:
                continue                                        # not a residual slot
            n_used += 1

            K_pool = (anchors_K + rk_val[ri]).flatten()         # [H_kv*D]

            # ground truth from the model's own weights, at this absolute position.
            # RoPE is ORTHOGONAL, so |rotated| == |unrotated| exactly -- scoring
            # against BOTH is what distinguished a pool/decoder frame disagreement
            # from a probe bug when they produced the same signature.
            try:
                K_true, K_true_raw = _gt(p)
            except Exception as e:                              # noqa: BLE001
                if n_fail == 0:
                    print(f"  [layer {li}] ground truth failed: "
                          f"{type(e).__name__}: {str(e)[:120]}")
                n_fail += 1
                continue
            if K_true is None:
                continue
            if K_true.shape != K_pool.shape:
                if n_fail == 0:
                    print(f"  [layer {li}] shape mismatch: pool "
                          f"{tuple(K_pool.shape)} vs true {tuple(K_true.shape)}")
                n_fail += 1
                continue

            a_ = K_pool / (K_pool.norm() + 1e-9)
            b_ = K_true / (K_true.norm() + 1e-9)
            r_ = K_true_raw / (K_true_raw.norm() + 1e-9)
            cos_l.append(float((a_ * b_).sum()))
            raw_l.append(float((a_ * r_).sum()))
            sin_l.append(float((K_pool - K_true).norm() / (K_true.norm() + 1e-9)))
            np_l.append(float(K_pool.norm()))
            nt_l.append(float(K_true.norm()))
            n_ok += 1

        if n_ok == 0:
            print(f"{li:>5}  no comparable slots "
                  f"(captured={len(cap[li])}, failed={n_fail})")
            continue
        # ── the LOW-RANK half: slots NOT in the residual set ──────────────────
        # The needle's own key is exact, so it can only lose the softmax if other
        # slots score spuriously high. Those are the ~half of each block rebuilt
        # from anchor + U@V. Under a ROTATED pool the SVD models post-RoPE deltas
        # whose phase wraps many times inside a 256-token block, so this is the
        # number that could have got worse exactly as the residual half got exact.
        U_q = pool.U[pidx].float().cpu() * float(pool.U_scale[pidx].item())   # [S,R]
        V_K = pool.V_KV[pidx, 0].float().cpu()                                # [R,H_kv,D]
        blk_scale = float(pool.scales[pidx].item())
        rk_set = set(x for x in rk_pos if x >= 0)
        lr_cos, lr_rel = [], []
        for off in range(0, slen):
            if off in rk_set:
                continue                                    # residual slots done above
            K_t, _ = _gt(anc + 1 + off)
            if K_t is None:
                continue
            dv = torch.einsum('r,rhd->hd', U_q[off], V_K) * blk_scale
            K_lr = (anchors_K + dv).reshape(-1)
            if K_lr.shape != K_t.shape:
                break
            lr_cos.append(float((K_lr / (K_lr.norm() + 1e-9)
                                 * K_t / (K_t.norm() + 1e-9)).sum()))
            lr_rel.append(float((K_lr - K_t).norm() / (K_t.norm() + 1e-9)))

        import statistics as st
        lr_c = st.mean(lr_cos) if lr_cos else float("nan")
        lr_r = st.mean(lr_rel) if lr_rel else float("nan")
        print(f"{li:>5} {n_used:>7} {st.mean(cos_l):>12.4f} "
              f"{st.mean(raw_l):>12.4f} {st.mean(sin_l):>9.4f} "
              f"{lr_c:>9.4f} {lr_r:>9.4f} {len(lr_cos):>6}")

    print("\nREAD THIS AS:")
    print("  Compare the column that MATCHES pool_stores_rotated_k() above.")
    print("  that column ~1.0  -> stored residuals are CORRECT; the loss is")
    print("     downstream of the pool (kernel read or sparse/dense merge).")
    print("  that column <<1   -> the residual VALUES are wrong.")
    print("  the OTHER column ~1.0 instead -> the pool's RoPE convention and the")
    print("     decoder's disagree: content is intact but stored in the wrong")
    print("     frame, which is a real runtime bug (identical norms, depressed")
    print("     cosine, worst in layers with the most rotary-range energy).")
    print("  BOTH well below 1.0 -> neither convention explains it; look further.")
    print("Compare --depth 0.9 (fails) against --depth 0.5 (passes): a number that")
    print("is the same in both does NOT explain why only one of them fails.")


if __name__ == "__main__":
    main()
