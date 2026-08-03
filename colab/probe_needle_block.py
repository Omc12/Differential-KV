"""Is the 32k@depth0.9 needle actually PRESERVED in the block that holds it?

32k@depth0.9 fails 0/3 with 'None' even with ALL 122 blocks attended and the
multi-chunk reduction eliminated (DKV_TOPK_BLOCKS=0 DKV_BLOCKS_PER_CHUNK=256).
Routing and reduction are therefore ruled out, and MLX recovers this same needle
from only ~4096 routed tokens. That makes it a CONTENT question, and content has
never been measured -- every round so far has measured selection.

This probe answers it WITHOUT needing a dense ground-truth K/V run, by asking
what the compressor decided:

  1. which block brackets the needle's tokens
  2. whether those tokens were chosen into the block's EXACT RESIDUAL set
  3. how much low-rank energy the block gives those slots

(2) is the decisive one. A block's residual budget is finite (max_residual);
compress_lowrank ranks tokens by a joint K/V error score and keeps the top ones.
If the needle's tokens lose that ranking contest at depth 0.9 but win it at
depth 0.0/0.5, the needle is being quantised away by the compressor and no
amount of routing or kernel work can recover it.

Depths 0.0 and 0.5 run as CONTROLS in the same process, same seed, same builder
as validate_cuda_dkv.py -- so the only variable is where the needle sits.

    python colab/probe_needle_block.py            # 32k, depths 0.0/0.5/0.9
    python colab/probe_needle_block.py --ctx 8k   # 8k sanity (all pass there)
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

NEEDLE = "ZEBRA-4471-QUARTZ"

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
    """Byte-identical to validate_cuda_dkv.build (same seed, same pool)."""
    filler = [random.choice(FILLER) for _ in range(n_filler)]
    at = int(len(filler) * depth)
    needle = (f"Remember this important code: {NEEDLE}. "
              "This is the only code you need to remember.")
    parts = filler[:at] + [needle] + filler[at:]
    parts.append("Question: What was the important code mentioned in this "
                 "text? Reply with only the code.")
    return " ".join(parts)


def needle_token_span(tok, prompt):
    """Absolute token indices covering the needle string."""
    ch = prompt.find(NEEDLE)
    if ch < 0:
        raise RuntimeError("needle string not found in prompt")
    try:
        enc = tok(prompt, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except (NotImplementedError, ValueError, KeyError):
        offsets = None                      # slow tokenizer: no offset mapping
    ids = tok(prompt).input_ids
    if offsets is not None:
        lo = hi = None
        for i, (a, b) in enumerate(offsets):
            if b <= ch or a >= ch + len(NEEDLE):
                continue
            lo = i if lo is None else lo
            hi = i
        if lo is not None:
            return lo, hi, len(ids)

    # Fallback: locate by decoding a sliding prefix. Slower but tokenizer-agnostic,
    # and correct even when the needle's tokens differ in isolation vs in context
    # (they do -- leading-space merges), which is why we do NOT encode NEEDLE alone
    # and search for that subsequence.
    lo = hi = None
    for i in range(len(ids)):
        if tok.decode(ids[:i + 1]).find(NEEDLE) >= 0:
            hi = i
            break
    if hi is None:
        raise RuntimeError("could not locate the needle in the token stream")
    for j in range(hi, -1, -1):
        if NEEDLE not in tok.decode(ids[j:hi + 1]):
            lo = j + 1
            break
    return (lo if lo is not None else hi), hi, len(ids)


def _pool_of(manager):
    for attr in ("pool", "block_pool", "native_pool"):
        p = getattr(manager, attr, None)
        if p is not None and hasattr(p, "U"):
            return p
    sm = getattr(manager, "_streaming_mgr", None)
    for attr in ("pool", "block_pool"):
        p = getattr(sm, attr, None)
        if p is not None and hasattr(p, "U"):
            return p
    raise RuntimeError("could not locate the NativeBlockPool on the manager")


def probe(w, ctx, label):
    import torch

    tok = w.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": ctx}],
                                     tokenize=False, add_generation_prompt=True)
    n_lo, n_hi, n_tok = needle_token_span(tok, prompt)
    print(f"\n─── {label} — {n_tok} tokens, needle at tokens {n_lo}..{n_hi} "
          f"({100.0 * n_lo / n_tok:.1f}% through) ───")

    # one short generation forces the full prefill + compression
    out = w.generate(prompt=prompt, max_new_tokens=8, temperature=0.0,
                     top_p=1.0, repetition_penalty=1.0)
    answer = out.rsplit("assistant", 1)[-1].strip().replace("\n", " ")
    got = "".join(c for c in answer.upper() if c.isalnum())
    want = "".join(c for c in NEEDLE.upper() if c.isalnum())
    print(f"  model answered: {answer[:70]!r}  -> "
          f"{'RECALLED' if want in got else 'MISSED'}")

    manager = w.manager
    pool = _pool_of(manager)
    sid = "default"

    layers = []
    for li in range(64):
        try:
            b = manager.get_streaming_blocks(sid, li)
        except Exception:
            b = None
        if b:
            layers.append((li, b))
    if not layers:
        print("  !! no streaming blocks found -- session id may differ")
        return

    for li, blocks in layers:
        hit = None
        for blk in blocks:
            a = getattr(blk, "anchor_idx", None)
            pidx = getattr(blk, "pool_idx", None)
            if a is None or pidx is None or pidx < 0:
                continue
            slen = int(pool.seq_lens[pidx].item())
            # anchor occupies a, active tokens are a+1 .. a+slen
            if a <= n_lo <= a + slen:
                hit = (blk, a, pidx, slen)
                break
        if hit is None:
            print(f"  layer {li:2d}: NEEDLE'S BLOCK NOT FOUND among "
                  f"{len(blocks)} blocks  <-- content is simply absent")
            continue

        blk, a, pidx, slen = hit
        # block-local offsets of the needle's tokens (0 == first ACTIVE token)
        off_lo, off_hi = n_lo - a - 1, n_hi - a - 1
        rk = pool.residual_K_positions[pidx]
        rv = pool.residual_V_positions[pidx]
        rk_set = set(int(x) for x in rk.tolist() if x >= 0)
        rv_set = set(int(x) for x in rv.tolist() if x >= 0)
        needle_offs = [o for o in range(off_lo, off_hi + 1) if 0 <= o < slen]
        in_k = sum(1 for o in needle_offs if o in rk_set)
        in_v = sum(1 for o in needle_offs if o in rv_set)

        # low-rank energy this block spends on the needle's slots vs its mean
        U = pool.U[pidx].float() * float(pool.U_scale[pidx].item())   # [S, R]
        eng = U.norm(dim=-1)                                          # [S]
        blk_mean = eng[:slen].mean().item() if slen else 0.0
        ndl_mean = (eng[needle_offs].mean().item() if needle_offs else 0.0)

        flag = "" if (needle_offs and in_k == len(needle_offs)) else "   <-- ***"
        print(f"  layer {li:2d}: block anchor={a} pool_idx={pidx} seq_len={slen} "
              f"| needle slots {needle_offs} "
              f"| residuals kept K={len(rk_set)} V={len(rv_set)} "
              f"| needle in residual set: K={in_k}/{len(needle_offs)} "
              f"V={in_v}/{len(needle_offs)} "
              f"| U-energy needle={ndl_mean:.4f} block-mean={blk_mean:.4f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32k", choices=("8k", "32k"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    a = ap.parse_args()

    # same serving config the validator asserts on, so this probes what ships
    try:
        from ACTIVE_RUNTIME.serving import decode_config
        for k, v in getattr(decode_config, "BEST_DECODE_DEFAULTS", {}).items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass
    os.environ.setdefault("DKV_SYNC_COMPRESS", "1")
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")

    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    w = PyTorchDKVHFWrapper(model_id=a.model, config={"mode": "fp16"},
                            device="cuda")
    w.ensure_loaded()

    n_filler = 800 if a.ctx == "8k" else 3200
    for depth in (0.0, 0.5, 0.9):
        random.seed(5)                      # validator's seed, identical filler
        probe(w, build(n_filler, depth), f"{a.ctx}@depth{depth:.1f}")

    print("\nREAD THIS AS:")
    print("  needle in residual set K=0/N  -> the COMPRESSOR dropped the needle;")
    print("     it lost the top-max_residual ranking contest inside its block.")
    print("     Nothing downstream can recover it. Fix belongs in compress_lowrank.")
    print("  needle in residual set K=N/N  -> content is preserved; the loss is")
    print("     downstream of the pool and this probe exonerates compression.")
    print("  block NOT FOUND               -> the span never reached the pool.")


if __name__ == "__main__":
    main()
