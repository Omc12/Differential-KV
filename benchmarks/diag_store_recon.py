#!/usr/bin/env python3
"""Runtime store vs exact K/V: what the decode path ACTUALLY rebuilds, per layer.

WHY. tier1_screen.py models the low-rank store offline and says Phi-4-mini's
layer 9 should be ~4% output error with the joint K|V factor (split no
better), while the runtime's decode-step layer diff measured 104% there. A
model of the store that disagrees with the store by 25x means the runtime
does something the model does not. This reads the runtime's own
reconstruction instead of a model of it.

It wraps remat_cache.reconstruct_blocks and the routed-block gather, so every
block the decode path materialises is recorded with its anchor position, then
compares it with the exact pre-RoPE K and V captured from the projections
during prefill (row 0 = the anchor at anchor_idx, row 1+j at anchor_idx+1+j).

    rel_err_K / rel_err_V   per block ||X_hat - X|| / ||X|| (raw, pre-RoPE)
    worst row               the token whose reconstruction is worst

USAGE
    python benchmarks/diag_store_recon.py --model microsoft/Phi-4-mini-reasoning \
        --layers 2 9 16 --ctx 32768
    DKV_KV_SPLIT=16,16 python benchmarks/diag_store_recon.py ...   (A/B)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE = os.path.join(ROOT, "ACTIVE_RUNTIME")
BENCH = os.path.join(ROOT, "benchmarks")


def _proj_widths(at, config):
    hd = int(getattr(at, "head_dim", 0) or config.hidden_size // config.num_attention_heads)
    kvh = int(getattr(at, "num_key_value_heads", 0) or config.num_key_value_heads)
    return hd, kvh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 9, 16])
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--new", type=int, default=2, help="decode tokens to generate")
    args = ap.parse_args()

    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")
    os.environ.setdefault("DKV_ROUTE_PROBE", "0")
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    import torch
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt
    from runtime.dkv_attention import resolve_decoder_layers
    import native_core.sparse_decode.remat_cache as RC
    import native_core.sparse_decode.triton_fused_decode as TF

    w = DKVHFWrapper(model_id=args.model, config={"quantization": "nf4", "preset": "mid"})
    w.ensure_loaded()
    tok, model = w.tokenizer, w.model
    layers = resolve_decoder_layers(model)
    want = set(args.layers)

    # ── exact K/V from the projections, prefill only ─────────────────────────
    exact = {li: {"k": [], "v": []} for li in want}
    cur = {"layer": -1, "decode": False}
    hooks = []
    for li in want:
        at = layers[li].self_attn
        hd, kvh = _proj_widths(at, model.config)
        nkv = hd * kvh
        if getattr(at, "qkv_proj", None) is not None and getattr(at, "k_proj", None) is None:
            nq = at.qkv_proj.out_features - 2 * nkv

            def h(_m, _i, out, li=li, nq=nq, nkv=nkv):
                if out.shape[1] > 1:
                    exact[li]["k"].append(out[0, :, nq:nq + nkv].detach().half().cpu())
                    exact[li]["v"].append(out[0, :, nq + nkv:].detach().half().cpu())
                elif "q" not in exact[li]:
                    # first decode step's query, pre-RoPE
                    exact[li]["q"] = out[0, -1, :nq].detach().float().cpu()
            hooks.append(at.qkv_proj.register_forward_hook(h))
        else:
            ksrc = getattr(at, "k_norm", None) or at.k_proj   # what attention rotates

            def hk(_m, _i, out, li=li):
                if out.shape[-2 if out.dim() == 4 else 1] > 1:
                    exact[li]["k"].append(out[0].reshape(out.shape[1], -1).detach().half().cpu())

            def hv(_m, _i, out, li=li):
                if out.shape[1] > 1:
                    exact[li]["v"].append(out[0].detach().half().cpu())

            qsrc = getattr(at, "q_norm", None) or at.q_proj

            def hq(_m, _i, out, li=li):
                # first decode step's query, pre-RoPE, flattened to [Hq * D]
                if out.shape[-2 if out.dim() == 4 else 1] == 1 and "q" not in exact[li]:
                    exact[li]["q"] = out[0].reshape(-1).detach().float().cpu()
            hooks += [ksrc.register_forward_hook(hk), at.v_proj.register_forward_hook(hv),
                      qsrc.register_forward_hook(hq)]

    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(
            lambda _m, _a, i=i: cur.__setitem__("layer", i)))

    # ── what decode materialises ─────────────────────────────────────────────
    seen = {}                       # layer -> {anchor_idx: (K [1+S,H,D], V)}
    global RES
    RES = {}                        # layer -> {anchor_idx: residual row set}
    last_anchor = {}
    _grb0, _rb0 = TF._gather_routed_blocks_for_kernel, RC.reconstruct_blocks

    def grb(pool, block_indices, anchor_indices, *a, **kw):
        last_anchor["a"] = anchor_indices
        return _grb0(pool, block_indices, anchor_indices, *a, **kw)

    def rb(*a, **kw):
        K, V = _rb0(*a, **kw)
        li = cur["layer"]
        anc = last_anchor.get("a")
        if li in want and anc is not None and len(anc) == K.shape[0]:
            d = seen.setdefault(li, {})
            rp = kw.get("res_pos")
            for n, ai in enumerate(anc.tolist()):
                if ai not in d:
                    d[ai] = (K[n].detach().float().cpu(), V[n].detach().float().cpu())
                    if rp is not None:
                        # residual rows are delta-row indices; +1 for the anchor row
                        RES.setdefault(li, {})[ai] = {int(x) + 1 for x in rp[n].tolist()
                                                      if int(x) >= 0}
        return K, V

    TF._gather_routed_blocks_for_kernel = grb
    RC.reconstruct_blocks = rb

    prompt = build_prompt(tok, args.ctx, args.depth)
    w.active_session = "store-recon"
    out = w.generate(prompt, max_new_tokens=args.new, temperature=0.0, top_p=1.0,
                     repetition_penalty=1.0,
                     query_text="What is the secret passcode? Repeat it exactly.")
    for h in hooks:
        h.remove()
    print(f"[recon] generated: {out!r}", flush=True)

    for li in sorted(want):
        K = torch.cat(exact[li]["k"]).float()
        V = torch.cat(exact[li]["v"]).float()
        P = K.shape[0]
        blocks = seen.get(li, {})
        if not blocks:
            print(f"layer {li}: no materialised block recorded")
            continue
        eKs, eVs = [], []
        worst = (0.0, None)
        for ai, (Kh, Vh) in sorted(blocks.items()):
            S1 = Kh.shape[0]
            if ai + S1 > P:
                continue
            Kx = K[ai:ai + S1].view_as(Kh)
            Vx = V[ai:ai + S1].view_as(Vh)
            # anchor-relative error: the anchor is exact, the store holds deltas
            dK, dV = Kx - Kx[:1], Vx - Vx[:1]
            eK = float((Kh - Kx).norm() / dK.norm().clamp_min(1e-9))
            eV = float((Vh - Vx).norm() / dV.norm().clamp_min(1e-9))
            eKs.append(eK)
            eVs.append(eV)
            row = ((Kh - Kx).flatten(1).norm(dim=1) / dK.flatten(1).norm(dim=1).clamp_min(1e-9))
            r = int(row.argmax())
            if float(row[r]) > worst[0]:
                worst = (float(row[r]), (ai, r))
        t = torch.tensor
        print(f"layer {li:>2}: {len(eKs)} blocks  rel err K median {t(eKs).median():.4f} "
              f"max {max(eKs):.4f}   V median {t(eVs).median():.4f} max {max(eVs):.4f}   "
              f"worst K row {worst[0]:.3f} at block {worst[1]}", flush=True)
        if "q" in exact[li]:
            _attention_check(li, exact[li]["q"], K, V, blocks, model, layers, P)


def _attention_check(li, q, K, V, blocks, model, layers, P):
    """Attention over the materialised blocks with the real first decode query:
    exact K/V vs the runtime's rebuilt K/V, both rotated at their own positions
    with the model's own rotary tables (attention_scaling included)."""
    import torch
    from runtime.dkv_attention import _apply_rope_single
    at = layers[li].self_attn
    hd, kvh = _proj_widths(at, model.config)
    hq = q.numel() // hd
    from runtime.dkv_attention import _resolve_rotary_emb
    rotary = _resolve_rotary_emb(model)            # the TEXT decoder's (not a vision tower's)
    dev = next(model.parameters()).device
    pos = torch.arange(P + 1, device=dev)[None]
    cos, sin = rotary(torch.zeros(1, 1, 1, device=dev), pos)
    cos, sin = cos[0].float().cpu(), sin[0].float().cpu()            # [P+1, rd]
    qr = _apply_rope_single(q.view(hq, hd), cos[P:P + 1], sin[P:P + 1])   # [Hq, D]
    Ke, Kh, Ve, Vh, ps = [], [], [], [], []
    for ai, (Khat, Vhat) in sorted(blocks.items()):
        S1 = Khat.shape[0]
        if ai + S1 > P:
            continue
        p = torch.arange(ai, ai + S1)
        Ke.append(K[ai:ai + S1].view(S1, kvh, hd))
        Ve.append(V[ai:ai + S1].view(S1, kvh, hd))
        Kh.append(Khat.view(S1, kvh, hd))
        Vh.append(Vhat.view(S1, kvh, hd))
        ps.append(p)
    p = torch.cat(ps)
    c, s = cos[p].unsqueeze(1), sin[p].unsqueeze(1)
    Ke = _apply_rope_single(torch.cat(Ke), c, s)
    Kh = _apply_rope_single(torch.cat(Kh), c, s)
    Ve, Vh = torch.cat(Ve), torch.cat(Vh)
    g = hq // kvh
    scale = float(getattr(at, "scaling", hd ** -0.5))
    kv_of = torch.arange(hq) // g
    le = torch.einsum("hd,nhd->hn", qr, Ke[:, kv_of]) * scale          # [Hq, N]
    lh = torch.einsum("hd,nhd->hn", qr, Kh[:, kv_of]) * scale
    pe, ph = le.softmax(-1), lh.softmax(-1)
    kl = (pe * (pe.clamp_min(1e-30).log() - ph.clamp_min(1e-30).log())).sum(-1)
    oe = torch.einsum("hn,nhd->hd", pe, Ve[:, kv_of])
    oh = torch.einsum("hn,nhd->hd", ph, Vh[:, kv_of])
    ok = torch.einsum("hn,nhd->hd", ph, Ve[:, kv_of])                  # keys' share
    rel = ((oh - oe).norm(dim=-1) / oe.norm(dim=-1).clamp_min(1e-9))
    relk = ((ok - oe).norm(dim=-1) / oe.norm(dim=-1).clamp_min(1e-9))
    # The same exact blocks through KIVI-4's quantizer, two ways: on POST-RoPE
    # keys (what a KIVI cache holds) and on PRE-RoPE keys rotated afterwards
    # (DKV's choice). If DKV's stored blocks are worse than both, the loss is
    # in DKV's store path, not in the 4-bit layout.
    def _kivi_k(x, G=32):                          # [N, H, D], per channel / G tokens
        n = x.shape[0] // G * G
        g = x[:n].float().view(-1, G, *x.shape[1:])
        lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
        s = (hi - lo).clamp_min(1e-8) / 15
        q = (torch.round((g - lo) / s).clamp(0, 15) * s + lo).view(n, *x.shape[1:])
        return torch.cat([q, x[n:].float()])
    def _kivi_v(x, G=32):                          # per token / G channels
        g = x.float().view(*x.shape[:-1], -1, G)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        s = (hi - lo).clamp_min(1e-8) / 15
        return (torch.round((g - lo) / s).clamp(0, 15) * s + lo).view(x.shape)
    Kraw = torch.cat([K[ai:ai + bk[0].shape[0]].view(-1, kvh, hd)
                      for ai, bk in sorted(blocks.items()) if ai + bk[0].shape[0] <= P])
    # The runtime (DKV and the KIVI baseline alike) holds rebuilt keys in the
    # model dtype; "+md" rounds the float rebuild there, before and after RoPE,
    # so a float32 reference cannot look better only for skipping that.
    md = next(model.parameters()).dtype
    md = md if md.is_floating_point else torch.bfloat16
    snap = lambda x: x.to(md).float()
    variants = {"kivi4 post-RoPE": (_kivi_k(Ke), _kivi_v(Ve)),
                "kivi4 pre-RoPE": (_apply_rope_single(_kivi_k(Kraw), c, s), _kivi_v(Ve)),
                "kivi4 post +md": (snap(_kivi_k(Ke)), snap(_kivi_v(Ve))),
                "kivi4 pre +md": (snap(_apply_rope_single(snap(_kivi_k(Kraw)), c, s)),
                                  snap(_kivi_v(Ve)))}
    for name, (Kq, Vq) in variants.items():
        lq = torch.einsum("hd,nhd->hn", qr, Kq[:, kv_of]) * scale
        pq = lq.softmax(-1)
        klq = (pe * (pe.clamp_min(1e-30).log() - pq.clamp_min(1e-30).log())).sum(-1)
        oq = torch.einsum("hn,nhd->hd", pq, Vq[:, kv_of])
        relq = ((oq - oe).norm(dim=-1) / oe.norm(dim=-1).clamp_min(1e-9))
        print(f"    {name:<16} attn KL mean {float(klq.mean()):.4f} max {float(klq.max()):.4f}"
              f"  out rel err {float(relq.mean()):.4f}", flush=True)
    # The same comparison against what the MODEL computes: exact keys rotated
    # in the model dtype (HF's apply_rotary_pos_emb casts cos/sin to it), DKV's
    # rebuilt pre-RoPE keys rotated the same way (the runtime's own rotation),
    # and KIVI quantizing those post-RoPE keys and dequantizing into the dtype.
    def _rope_md(x):
        return _apply_rope_single(x.to(md), c.to(md), s.to(md)).float()
    Ke_m = _rope_md(torch.cat([K[ai:ai + bk[0].shape[0]].view(-1, kvh, hd)
                               for ai, bk in sorted(blocks.items()) if ai + bk[0].shape[0] <= P]))
    Kh_m = _rope_md(torch.cat([bk[0].view(-1, kvh, hd)
                               for ai, bk in sorted(blocks.items()) if ai + bk[0].shape[0] <= P]))
    pe_m = (torch.einsum("hd,nhd->hn", qr, Ke_m[:, kv_of]) * scale).softmax(-1)
    for name, Kq in (("dkv (md rope)", Kh_m), ("kivi4 post (md)", snap(_kivi_k(Ke_m))),
                     ("kivi4 pre (md)", _rope_md(_kivi_k(Kraw)))):
        pq = (torch.einsum("hd,nhd->hn", qr, Kq[:, kv_of]) * scale).softmax(-1)
        klq = (pe_m * (pe_m.clamp_min(1e-30).log() - pq.clamp_min(1e-30).log())).sum(-1)
        print(f"    vs md-ref {name:<16} attn KL mean {float(klq.mean()):.4f} "
              f"max {float(klq.max()):.4f}", flush=True)
    h = int(kl.argmax())
    dl = (lh[h] - le[h])
    top = torch.topk(dl.abs(), 3).indices.tolist()
    res = RES.get(li, {})
    anc_of = {}
    for ai, (Khat, _v) in sorted(blocks.items()):
        for r in range(Khat.shape[0]):
            anc_of[ai + r] = (ai, r)

    def _isres(pp):
        ai, r = anc_of.get(pp, (None, None))
        return "res" if ai is not None and r in res.get(ai, set()) else "lowrank"
    rows = ", ".join(f"pos {int(p[i])} [{_isres(int(p[i]))}] dlogit {float(dl[i]):+.2f} p_exact {float(pe[h, i]):.3f}"
                     f" p_hat {float(ph[h, i]):.3f}" for i in top)
    # Where the worst-logit rows rank in their own block by stored error.
    for i in top[:2]:
        pp = int(p[i])
        ai, r = anc_of[pp]
        Khat, Vhat = blocks[ai]
        S1 = Khat.shape[0]
        Kx, Vx = K[ai:ai + S1].view_as(Khat), V[ai:ai + S1].view_as(Vhat)
        eK = (Khat - Kx).flatten(1).norm(dim=1)
        eV = (Vhat - Vx).flatten(1).norm(dim=1)
        dK = (Kx - Kx[:1]).flatten(1).norm(dim=1)
        dV = (Vx - Vx[:1]).flatten(1).norm(dim=1)
        rk = int((eK > eK[r]).sum()) + 1
        rv = int((eV > eV[r]).sum()) + 1
        print(f"    pos {pp}: |dK| {float(dK[r]):.2f} (block median {float(dK.median()):.2f})"
              f"  |eK| {float(eK[r]):.2f} rank {rk}/{S1}   |dV| {float(dV[r]):.3f} "
              f"(median {float(dV.median()):.3f})  |eV| {float(eV[r]):.3f} rank {rv}"
              f"   smallest |eK| among residual rows "
              f"{min(float(eK[j]) for j in res.get(ai, {0})):.2f}", flush=True)
    nres = [len(v) for v in res.values()]
    print(f"    residual rows per block: {nres[:4]}... ", flush=True)
    print(f"    attn over {Ke.shape[0]} rows: KL mean {float(kl.mean()):.4f} max {float(kl.max()):.4f}"
          f" (head {h})  out rel err mean {float(rel.mean()):.4f}  (keys alone "
          f"{float(relk.mean()):.4f})\n    head {h} worst logits: {rows}", flush=True)


if __name__ == "__main__":
    main()
