"""Is it the QUERY? Score the needle's exact key with DKV's q and with dense's q.

WHY THIS IS THE REMAINING QUESTION
----------------------------------
For 32k@depth0.9 every key-side input is now verified:
  * the needle's stored key is EXACT — probe_residual_values measures
    anchors_K + residual_K_values against RoPE(k_proj(h),pos) at cos 1.0000,
    rel_err 3e-4, at every layer, IDENTICALLY on the failing 0.9 and passing 0.5;
  * its block is routed at rank 0-1 of 16, its row is unmasked, its offset resolves;
  * the sparse-half math is MLX-equivalent (delta_s + s_anc, twins masked, 1/sqrt(D));
  * the merge is NOT the cause: with the MLX partition and a genuinely adaptive
    `auto`, 32k@0.9 did not move, and remat (one unbiased softmax, no merge at all)
    fails identically while MLX passes 9/9 at its own 0.0 default.

So in a single honest softmax the needle's row carries a correct key, a correct
mask and a correct index, and still takes ~1e-3 while the newest dense rows take
up to 0.69. The only input left is q.

WHAT IT MEASURES
    score = (q . k_true) / sqrt(D)   for the needle's tokens,
with k_true = RoPE(k_proj(h_p), p) built from the model's own weights — the same
ground truth probe_residual_values validated the pool against — and q taken at the
decode step that must emit the answer, once from DKV and once from dense.

Dense's q is the control: it is the query that DOES retrieve the needle (the dense
run passes all nine cases). Scoring BOTH queries against the SAME key isolates q,
because the key is held fixed by construction.

    python colab/probe_query_vs_dense.py --depth 0.9 --mode dkv     # writes a cache
    python colab/probe_query_vs_dense.py --depth 0.9 --mode dense   # prints the table
    python colab/probe_query_vs_dense.py --depth 0.5 --mode dkv     # PASSING control
    python colab/probe_query_vs_dense.py --depth 0.5 --mode dense

Two passes, not two models in one process: a 32k KV cache plus two copies of the
weights does not fit in 7.81 GiB, and the earlier MLX probe died exactly this way.

READ IT AS
  DKV's score far below dense's, and only at 0.9 -> the query is degraded. The
     whole KV path is then exonerated and the defect is upstream in the hidden
     states, which would explain why eight rounds of KV-side changes were
     byte-identical.
  The two agree at both depths -> the query is fine and the key is exact, so the
     needle genuinely is not what the model reaches for; look at the value side
     or the decoder.
  Compare against depth 0.5 EVERY TIME. A number that is the same on the passing
  depth does not explain why only one of them fails — the trap this investigation
  has already hit twice.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

from probe_needle_block import (          # noqa: E402  (path set above)
    NEEDLE, build, needle_token_span,
)

CACHE = "/tmp/dkv_qprobe_{mode}_{depth}.pt"


def _ctx(depth):
    """Replay the validator's FULL nine-case RNG sequence — it seeds once, so a
    case built in isolation is a DIFFERENT PROMPT."""
    random.seed(5)
    ctx = None
    cases = [("2k", 200, d) for d in (0.0, 0.5, 0.9)]
    cases += [("8k", 800, d) for d in (0.0, 0.5, 0.9)]
    cases += [("32k", 2400, d) for d in (0.0, 0.5, 0.9)]
    for lbl, n, d in cases:
        c = build(n, d)
        if lbl == "32k" and abs(d - depth) < 1e-9:
            ctx = c
    assert ctx is not None, f"no 32k case at depth {depth}"
    return ctx


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.9)
    ap.add_argument("--mode", choices=("dkv", "dense", "compare"), default="dkv")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--max-new", type=int, default=24)
    a = ap.parse_args()

    if a.mode == "compare":
        # Re-analyse cached runs. No model load, no GPU -- the caches already hold
        # EVERY decode step, and the first report read the wrong one.
        _report(torch.load(CACHE.format(mode="dkv", depth=a.depth), weights_only=False),
                torch.load(CACHE.format(mode="dense", depth=a.depth), weights_only=False),
                a.depth)
        return

    try:
        from ACTIVE_RUNTIME.serving import decode_config
        for k, v in getattr(decode_config, "BEST_DECODE_DEFAULTS", {}).items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass
    os.environ.setdefault("DKV_SYNC_COMPRESS", "1")
    if a.mode == "dense":
        # Same wrapper, same weights, DKV never engages — so the ONLY difference
        # between the two passes is the compressed KV path itself. Loading a
        # separate plain-HF model would also change the code path around it.
        os.environ["DKV_ENGAGE_THRESHOLD"] = "100000000"
    else:
        os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")

    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    w = PyTorchDKVHFWrapper(model_id=a.model, config={"mode": "fp16"}, device="cuda")
    w.ensure_loaded()

    tok = w.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": _ctx(a.depth)}],
                                     tokenize=False, add_generation_prompt=True)
    n_lo, n_hi, n_tok = needle_token_span(tok, prompt)
    print(f"\n=== 32k@depth{a.depth} [{a.mode}] — {n_tok} tok, "
          f"needle at {n_lo}..{n_hi} ===")

    model = w.model if hasattr(w, "model") else w._model
    layers = model.model.layers
    attn_layers = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]

    # Capture the needle's hidden states (for k_true) AND every decode step's
    # hidden state (for q). Decode steps are the T==1 forwards after prefill.
    cap_k = {i: {} for i in attn_layers}      # layer -> {abs_pos: h}   (needle)
    cap_q = {i: [] for i in attn_layers}      # layer -> [h] per decode step
    seen = {i: 0 for i in attn_layers}

    # INPUT hidden state to EVERY decoder layer (not just the attention ones) at
    # each decode step. Layer 3 is the first DKV layer and its input comes from
    # layers 0-2, which are linear-attention and never touch the compressed KV --
    # so at depth 0.5 DKV and dense agree there to 1.0000, as they must. At 0.9
    # they do not, which puts the divergence UPSTREAM of everything this
    # investigation has been looking at. This finds the first layer that differs.
    cap_h = {i: [] for i in range(len(layers))}
    seen_h = {i: 0 for i in range(len(layers))}

    def mk_hook_h(li):
        def hook(mod, args, kwargs):
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            if hs is None or not hasattr(hs, "dim") or hs.dim() != 3:
                return
            start = seen_h[li]
            T = hs.shape[1]
            seen_h[li] = start + T
            if T == 1 and start >= n_tok:
                # `start` is recorded so the comparison can VERIFY the two runs are
                # at the same absolute position instead of assuming index k means
                # the same step in both. It does not: the runs emit different token
                # counts once they diverge, and one may feed the final prompt token
                # as a T==1 forward where the other does not. Layer 0's input is the
                # token embedding, so a mismatched pair there reads as cos~0.03 --
                # which looks like a spectacular finding and is only misalignment.
                cap_h[li].append((start, hs[0, 0].detach().float().cpu()))
        return hook

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
            if T == 1 and start >= n_tok:            # a decode step
                cap_q[li].append((start, hs[0, 0].detach().float().cpu()))
                return
            lo, hi = max(n_lo, start), min(n_hi, start + T - 1)
            for p in range(lo, hi + 1):
                cap_k[li][p] = hs[0, p - start].detach().float().cpu()
        return hook

    handles = [layers[i].self_attn.register_forward_pre_hook(mk_hook(i),
                                                             with_kwargs=True)
               for i in attn_layers]
    handles += [layers[i].register_forward_pre_hook(mk_hook_h(i), with_kwargs=True)
                for i in range(len(layers))]
    try:
        out = w.generate(prompt=prompt, max_new_tokens=a.max_new, temperature=0.0,
                         top_p=1.0, repetition_penalty=1.0)
    finally:
        for h in handles:
            h.remove()

    ans = out.rsplit("assistant", 1)[-1].strip().replace("\n", " ")
    got = "".join(c for c in ans.upper() if c.isalnum())
    want = "".join(c for c in NEEDLE.upper() if c.isalnum())
    recalled = want in got
    print(f"answered {ans[:60]!r} -> {'RECALLED' if recalled else 'MISSED'}")

    # ── q and k_true from the model's own weights ────────────────────────────
    rot = model.model.rotary_emb
    dev = model.device

    from ACTIVE_RUNTIME.native_core.sparse_decode.triton_fused_decode import (
        _partial_rope_apply,
    )

    def _rope(x, p):
        """[1, H, 1, D] at absolute position p, via the model's own rotary_emb.

        Mirrors probe_residual_values._gt exactly, including the fp32 cast BEFORE
        the rotation — that probe is the one whose output was validated against
        ground truth at cos 1.0000, so any deviation here would be measuring a
        different quantity than the one already trusted.
        """
        c, s = rot(x, torch.tensor([[p]], device=dev))
        return _partial_rope_apply(x.float(), c.float().unsqueeze(1),
                                   s.float().unsqueeze(1))

    rows = {}
    for li in attn_layers:
        attn = layers[li].self_attn
        dt = next(attn.parameters()).dtype
        Dh = getattr(attn, "head_dim", None) or model.config.head_dim

        # k_true for each needle token (ground truth; identical in both passes up
        # to the hidden states each run actually produced).
        ks = []
        for p in range(n_lo, n_hi + 1):
            if p not in cap_k[li]:
                continue
            h = cap_k[li][p].to(dev, dtype=dt)
            kt = attn.k_proj(h.unsqueeze(0))
            if getattr(attn, "k_norm", None) is not None:
                kt = attn.k_norm(kt.view(1, -1, Dh))
            kt = kt.view(1, 1, -1, Dh).transpose(1, 2)          # [1,H_kv,1,D]
            ks.append((p, _rope(kt, p).float().cpu()))

        # q at every decode step.
        qs = []
        for (p, h) in cap_q[li]:
            hh = h.to(dev, dtype=dt)
            qt = attn.q_proj(hh.unsqueeze(0))
            if getattr(attn, "q_norm", None) is not None:
                qt = attn.q_norm(qt.view(1, -1, Dh))
            qt = qt.view(1, 1, -1, Dh).transpose(1, 2)          # [1,H_q,1,D]
            qs.append((p, _rope(qt, p).float().cpu()))
        rows[li] = {"k": ks, "q": qs, "Dh": Dh}

    payload = {"rows": rows, "recalled": recalled, "depth": a.depth,
               "n_lo": n_lo, "n_hi": n_hi, "answer": ans[:60],
               "hidden": {i: v for i, v in cap_h.items()}}   # [(abs_pos, h), ...]
    path = CACHE.format(mode=a.mode, depth=a.depth)
    torch.save(payload, path)
    print(f"[saved] {path}")

    other = CACHE.format(mode=("dense" if a.mode == "dkv" else "dkv"),
                         depth=a.depth)
    if not os.path.exists(other):
        print(f"\nNow run the other pass:\n  python {sys.argv[0]} "
              f"--depth {a.depth} --mode "
              f"{'dense' if a.mode == 'dkv' else 'dkv'}")
        return
    _report(torch.load(CACHE.format(mode="dkv", depth=a.depth), weights_only=False),
            torch.load(CACHE.format(mode="dense", depth=a.depth), weights_only=False),
            a.depth)


def _report(dkv, dense, depth):
    import torch
    print(f"\n=== q COMPARISON @ depth {depth} ===")
    print(f"  DKV   : {'RECALLED' if dkv['recalled'] else 'MISSED'}  {dkv['answer']!r}")
    print(f"  dense : {'RECALLED' if dense['recalled'] else 'MISSED'}  {dense['answer']!r}")
    print("\nBest needle score over its tokens, at the LAST decode step, per layer.")
    print("k_true is DENSE's (the query that provably retrieves it), so both q's")
    print("are scored against the SAME key -- that is what isolates q.\n")
    # PER STEP, not just the last. The two runs share the '<think>\n\n</think>\n\n'
    # prefix, so the EARLY steps have IDENTICAL token history and the comparison
    # there is clean. By the last step DKV has emitted 'None' and dense the code,
    # so their hidden states differ BECAUSE the answers differ -- reading that step
    # cannot separate "q degraded -> wrong answer" from "wrong answer -> different
    # history -> different q". Step 0 can.
    for li in sorted(dkv["rows"]):
        d_, n_ = dkv["rows"][li], dense["rows"][li]
        if not d_["q"] or not n_["q"] or not n_["k"]:
            continue
        Dh = n_["Dh"]
        sc = 1.0 / (Dh ** 0.5)
        print(f"\n  layer {li}")
        print(f"  {'step':>5} {'score_DKV':>12} {'score_dense':>12} {'ratio':>8} "
              f"{'cos(q_dkv,q_dense)':>20}")
        n_steps = min(len(d_["q"]), len(n_["q"]))
        for si in range(n_steps):
            qd, qn = d_["q"][si][1], n_["q"][si][1]
            H_q = qd.shape[1]
            best_d = best_n = -1e30
            for (_, kt) in n_["k"]:                      # dense's k_true, both times
                H_kv = kt.shape[1]
                rep = max(1, H_q // H_kv)
                k = kt.repeat_interleave(rep, dim=1)
                best_d = max(best_d, float((qd * k).sum(-1).max()) * sc)
                best_n = max(best_n, float((qn * k).sum(-1).max()) * sc)
            cq = torch.nn.functional.cosine_similarity(
                qd.flatten(), qn.flatten(), dim=0).item()
            ratio = best_d / best_n if abs(best_n) > 1e-9 else float("nan")
            print(f"  {si:>5} {best_d:>12.4f} {best_n:>12.4f} {ratio:>8.3f} "
                  f"{cq:>20.4f}")
    # ── Where does the hidden state FIRST diverge? ───────────────────────────
    hd, hn = dkv.get("hidden"), dense.get("hidden")
    if hd and hn:
        print("\n=== INPUT hidden state per DECODER layer, decode step 0 ===")
        print("Layer 0's input is the embedding: it MUST be 1.0 (same prompt, same")
        print("tokens). The first layer below 1.0 is where DKV starts to differ --")
        print("and anything before layer 3 is upstream of the compressed-KV path")
        print("entirely, since layers 0-2 are linear-attention and never touch it.\n")
        # Align on ABSOLUTE POSITION, never on list index. Layer 0 must come out
        # 1.0000 (same token, same embedding table); if it does not, the pair is
        # misaligned and NOTHING below it can be read.
        pos_d = {p for (p, _) in hd.get(0, [])}
        pos_n = {p for (p, _) in hn.get(0, [])}
        common = sorted(pos_d & pos_n)
        if not common:
            print("  NO COMMON DECODE POSITION between the runs — cannot compare.")
            return
        at = common[0]
        print(f"  comparing at absolute position {at} "
              f"(DKV has {len(pos_d)} decode steps, dense {len(pos_n)})")
        print(f"  {'layer':>5} {'cos':>10} {'rel_err':>10}")
        first_bad = None
        for li in sorted(hd):
            x = dict(hd.get(li, [])).get(at)
            y = dict(hn.get(li, [])).get(at)
            if x is None or y is None:
                continue
            c = torch.nn.functional.cosine_similarity(
                x.flatten(), y.flatten(), dim=0).item()
            rel = (torch.norm(x - y) / torch.clamp(torch.norm(y), min=1e-9)).item()
            if first_bad is None and c < 0.9999:
                first_bad = li
            print(f"  {li:>5} {c:>10.4f} {rel:>10.4f}"
                  f"{'   <-- FIRST DIVERGENCE' if first_bad == li else ''}")
        if first_bad == 0:
            print("\n  ⚠ LAYER 0 DIVERGED. Its input is the token embedding, so with")
            print("  the same token this is IMPOSSIBLE -- the two runs are still")
            print("  misaligned (different token fed at this position). Treat every")
            print("  row above as INVALID rather than as a result.")
        elif first_bad is not None:
            print(f"\nFirst divergence at decoder layer {first_bad}. "
                  f"{'A LINEAR-ATTENTION layer -- upstream of the compressed KV entirely.' if first_bad < 3 else 'At/after the first full-attention layer, i.e. the KV path.'}")

    print("\nSTEP 0 IS THE ONE THAT MATTERS: identical history in both runs, so a")
    print("low cos THERE is causal. A cos that starts ~1.0 and only falls at later")
    print("steps is the answers having diverged -- an effect, not the cause.")
    print("\nDKV far below dense here, and only at 0.9 -> the QUERY is degraded;")
    print("the KV path is exonerated and the defect is upstream in the hidden states.")
    print("The two agreeing at BOTH depths -> q is fine and the key is exact, so")
    print("the needle is genuinely not what the model reaches for (value side or")
    print("decoder). Always read 0.9 against the 0.5 control.")


if __name__ == "__main__":
    main()
