#!/usr/bin/env python3
"""Tier 1: screen store variants offline, in seconds, on captured real KV.

Reads tier1_capture.py dumps (pre-RoPE K, V, decode queries) and, for each
variant, rebuilds the compressed store the way DKV does -- per 1024-token
block an exact anchor, a factorization of the block's offset from it across
all KV heads and both K and V, int-quantized U, 16-bit V factor, a budget of
8-bit residual rows chosen per block, and an exact dense window of the most
recent tokens -- then attends with the REAL decode queries and compares the
attention output to exact attention:

    out_err   mean ||o_hat - o|| / ||o|| over heads, decode steps, layers
    attn_kl   mean KL(p || p_hat) of the attention distribution
    B/tok     stored bytes per prompt token (store + window), and cmp against
              a 16-bit dense cache

It is a model of the store, not the runtime: no routing (every block is
attended, as at these lengths), exact SVD where the runtime runs a
randomized one, and none of the runtime's content heuristics (token boosts).
Read variants against `base` here; tier 2 checks the winners in real decode.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

BLOCK = 1024


# ───────────────────────────────────────────────────────────── helpers ──
def rope_tables(rec, P, device):
    """RoPE tables and attention scale as the capture read them off the model."""
    return (rec["cos"][:P].to(device), rec["sin"][:P].to(device), None,
            int(rec["head_dim"]), float(rec["scale"]))


def rot(x, cos, sin):
    """x [P, H, D], cos/sin [P, rd] -- HF rotate_half convention. When the
    tables are narrower than D (partial rotary), only the first rd dims rotate."""
    rd = cos.shape[-1]
    xr_, xp = x[..., :rd], x[..., rd:]
    d = rd // 2
    xr = torch.cat([-xr_[..., d:], xr_[..., :d]], -1)
    out = xr_ * cos[:, None, :] + xr * sin[:, None, :]
    return torch.cat([out, xp], -1) if xp.shape[-1] else out


def quant_rows(x, bits, group):
    """Asymmetric per-row, per-channel-group fake quantization."""
    if bits >= 16:
        return x.half().float()
    n, f = x.shape
    g = x.view(n, f // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    q = (2 ** bits) - 1
    s = (hi - lo).clamp(min=1e-8) / q
    return (torch.round((g - lo) / s).clamp(0, q) * s + lo).view(n, f)


def svd_r(X, r):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U[:, :r], S[:r], Vh[:r]


def factor(X, r, chnorm):
    """Token-norm-normalized truncated SVD; returns U_scaled [n,r], Vh [r,f]."""
    c = None
    if chnorm:
        c = X.pow(2).mean(0).sqrt().clamp(min=1e-6)
        X = X / c
    tn = X.norm(dim=1).clamp(min=1e-5)
    U, S, Vh = svd_r(X / tn[:, None], r)
    Us = U * S * tn[:, None]
    if c is not None:
        Vh = Vh * c
    return Us, Vh


# ───────────────────────────────────────────────────────────── store ──
def compress_block(K, V, P, qstat, attn_w):
    """K, V [n, H*D] (anchor row first). Returns (K_hat, V_hat, bytes)."""
    n, hd_all = K.shape
    aK, aV = K[:1], V[:1]
    dK, dV = K[1:] - aK, V[1:] - aV
    m = dK.shape[0]
    g = 1.0
    if P["split"]:
        rK, rV = P["split"]
        UK, VhK = factor(dK, rK, P["chnorm"])
        UV, VhV = factor(dV, rV, P["chnorm"])
        U = torch.cat([UK, UV], 1)
        Vh = torch.zeros(rK + rV, 2 * hd_all, device=K.device)
        Vh[:rK, :hd_all] = VhK
        Vh[rK:, hd_all:] = VhV
    else:
        X = torch.cat([dK, dV], 1)
        if P["vgain"]:
            g = float((dK.pow(2).sum() / dV.pow(2).sum().clamp(min=1e-12)).sqrt().clamp(1, 1e4))
        Xs = torch.cat([dK, dV * g], 1)
        U, Vh = factor(Xs, P["rank"], P["chnorm"])
        Vh = torch.cat([Vh[:, :hd_all], Vh[:, hd_all:] / g], 1)
    r = U.shape[1]
    # U quantization (per block, optionally per column) and a 16-bit V factor
    if P["u_colscale"]:
        s = U.abs().amax(0).clamp(min=1e-8)
        U, Vh = U / s, Vh * s[:, None]
    qmax = 2 ** (P["u_bits"] - 1) - 1
    us = U.abs().max().clamp(min=1e-8) / qmax
    U = torch.round(U / us).clamp(-qmax, qmax) * us
    Vh = Vh.half().float()
    rec = U @ Vh
    rK_, rV_ = rec[:, :hd_all], rec[:, hd_all:]
    eK, eV = dK - rK_, dV - rV_
    relK = eK.norm(dim=1) / dK.norm(dim=1).clamp(min=1e-8)
    relV = eV.norm(dim=1) / dV.norm(dim=1).clamp(min=1e-8)
    RK, RV = P["resK"], P["resV"]
    if P["tiers"]:
        tail = max(float(relK.median()), float(relV.median()))
        cap = 8 if tail < 0.05 else (16 if tail < 0.15 else None)
        if cap is not None:
            RK, RV = min(RK, cap), min(RV, cap)
    joint_set = P["score"] in ("joint", "jointattn")
    if joint_set:
        # The runtime's rule: ONE index set for K and V, ranked by the joint
        # ABSOLUTE error with V balanced by the same gain the SVD used.
        gb = g if (not P["split"] and P["vgain"]) else 1.0
        sK = (eK.pow(2).sum(1) + (gb * eV).pow(2).sum(1)).sqrt()
        if P["score"] == "jointattn":
            sK = sK * (attn_w[1:] / attn_w[1:].mean().clamp(min=1e-12))
        sV = sK
    elif P["score"] == "rel":
        sK, sV = relK, relV
    elif P["score"] == "qenergy":
        # expected squared logit error under prompt queries, RoPE-averaged:
        # sum over rotation pairs of E[|q_pair|^2] * |e_pair|^2
        H = qstat.shape[0]
        D = hd_all // H
        e = eK.view(m, H, D)
        ep = e[..., :D // 2] ** 2 + e[..., D // 2:] ** 2           # [m, H, D/2]
        sK = (ep * qstat[None]).sum((1, 2)).sqrt()
        sV = eV.norm(dim=1) * (attn_w[1:] if attn_w is not None else 1.0)
    elif P["score"] in ("attnw", "attnwV"):
        w = attn_w[1:] / attn_w[1:].mean().clamp(min=1e-12)
        sK = relK if P["score"] == "attnwV" else eK.norm(dim=1) * w
        sV = eV.norm(dim=1) * w
    else:
        raise ValueError(P["score"])
    if P.get("_R") is not None:                  # block-adaptive budget (B3)
        RK = RV = P["_R"]
    Kh, Vhh = aK + rK_, aV + rV_
    tb = P.get("tok_bits", 0)
    if tb:
        # B1: every row gets a low-bit copy of what the factorization missed
        # (GEAR-style), so near-duplicates stay distinguishable.
        G = P.get("tok_group", 64)
        Kh = aK + rK_ + quant_rows(eK, tb, G)
        Vhh = aV + rV_ + quant_rows(eV, tb, G)
    selK = selV = None
    if RK > 0:
        selK = torch.topk(sK, min(RK, m)).indices
        Kh[selK] = aK + quant_rows(dK[selK], P["res_bits"], 64)
    if RV > 0:
        selV = torch.topk(sV, min(RV, m)).indices
        Vhh[selV] = aV + quant_rows(dV[selV], P["res_bits"], 64)
    # bytes: anchor + U + U scales + V factor + residual slots (budget, as the
    # pool preallocates them uniformly) + per-token low-bit residuals
    res_row = hd_all * P["res_bits"] / 8 + (hd_all / 64 * 4 if P["res_bits"] < 16 else 0) + 2
    Rb = (P["_R"] * 2) if P.get("_R") is not None else (P["resK"] + P["resV"])
    b = (2 * hd_all * 2 + m * r * P["u_bits"] / 8 + (4 * r if P["u_colscale"] else 4)
         + r * 2 * hd_all * 2 + Rb * res_row)
    if tb:
        G = P.get("tok_group", 64)
        b += 2 * m * (hd_all * tb / 8 + hd_all / G * 4)
    return torch.cat([aK, Kh]), torch.cat([aV, Vhh]), b


def block_error_energy(K, V, P):
    """Joint error energy left by the factorization in one block (for B3)."""
    P0 = dict(P, resK=0, resV=0, tok_bits=0, tiers=False, _R=None)
    n, hd = K.shape
    Kh, Vh, _ = compress_block(K, V, P0, None, None)
    return float((K - Kh).pow(2).sum() + (V - Vh).pow(2).sum())


def build_kivi(K, V, T, bits, group=32, resid=128):
    """KIVI reference: keys per channel over groups of `group` tokens, values
    per token, `bits`-bit asymmetric; the last `resid` tokens in 16 bits."""
    Kh, Vh = K.clone(), V.clone()
    hd = K.shape[1]
    Q = max(0, ((T - resid) // group) * group)
    q = 2 ** bits - 1
    for g0 in range(0, Q, group):
        blk = K[g0:g0 + group]
        lo, hi = blk.amin(0, keepdim=True), blk.amax(0, keepdim=True)
        s = (hi - lo).clamp(min=1e-8) / q
        Kh[g0:g0 + group] = torch.round((blk - lo) / s).clamp(0, q) * s + lo
    D = 128 if hd % 128 == 0 else hd
    Vh[:Q] = quant_rows(V[:Q], bits, D)
    nb = (Q * hd * bits / 8 + (Q / group) * hd * 4          # K codes + per-channel scale/zero
          + Q * hd * bits / 8 + Q * (hd / D) * 4            # V codes + per-token scale/zero
          + (T - Q) * 2 * hd * 2)                           # 16-bit recent window
    return Kh, Vh, nb, Q


def lr_side(X, r, R, u_bits=8, res_bits=8, tok_bits=0, tok_group=64):
    """Low-rank store for ONE side (K or V) of the blocks: anchor + factor of
    the offsets (r columns) + top-R 8-bit residual rows (+ optional per-token
    low-bit residual). Returns (X_hat, bytes) for X [n, F] (anchor first)."""
    a, d = X[:1], X[1:] - X[:1]
    m, F = d.shape
    Us, Vh = factor(d, r, False)
    s = Us.abs().amax(0).clamp(min=1e-8)
    Us, Vh = Us / s, Vh * s[:, None]
    qm = 2 ** (u_bits - 1) - 1
    Us = torch.round(Us * qm).clamp(-qm, qm) / qm
    rec = Us @ Vh.half().float()
    e = d - rec
    Xh = a + rec
    if tok_bits:
        Xh = Xh + quant_rows(e, tok_bits, tok_group)
    if R > 0:
        i = torch.topk(e.norm(dim=1), min(R, m)).indices
        Xh[i] = a + quant_rows(d[i], res_bits, 64)
    b = (F * 2 + m * r * u_bits / 8 + 4 * r + r * F * 2
         + R * (F * res_bits / 8 + F / 64 * 4 + 2))
    if tok_bits:
        b += m * (F * tok_bits / 8 + F / tok_group * 4)
    return torch.cat([a, Xh]), b


def build_hybrid(K, V, T, P):
    """Per-side choice: keys and values each stored low-rank ('lr') or
    KIVI-quantized ('kq<bits>'), with DKV's dense window."""
    W = P["window"]
    C = max(0, ((T - W) // BLOCK) * BLOCK)
    Kh, Vh = K.clone(), V.clone()
    nb = 0.0
    for X, Xh, side in ((K, Kh, P["hK"]), (V, Vh, P["hV"])):
        if side.startswith("kt"):
            # per-TOKEN quantization in channel groups (the residual store's
            # layout): "kt4g64" = 4 bits, groups of 64 channels
            bits, grp = side[2:].split("g")
            bits, grp = int(bits), int(grp)
            Xh[:C] = quant_rows(X[:C], bits, grp)
            nb += C * X.shape[1] * bits / 8 + C * (X.shape[1] / grp) * 4
            continue
        if side.startswith("kq"):
            bits = int(side[2:])
            if X is K:
                q = 2 ** bits - 1
                for g0 in range(0, C, 32):
                    blk = X[g0:g0 + 32]
                    lo, hi = blk.amin(0, keepdim=True), blk.amax(0, keepdim=True)
                    s = (hi - lo).clamp(min=1e-8) / q
                    Xh[g0:g0 + 32] = torch.round((blk - lo) / s).clamp(0, q) * s + lo
                nb += C * X.shape[1] * bits / 8 + (C / 32) * X.shape[1] * 4
            else:
                Xh[:C] = quant_rows(X[:C], bits, 128)
                nb += C * X.shape[1] * bits / 8 + C * (X.shape[1] / 128) * 4
        else:
            for b0 in range(0, C, BLOCK):
                xb, b = lr_side(X[b0:b0 + BLOCK], P["hr"], P["hR"], P["u_bits"],
                                tok_bits=P.get("htok", 0))
                Xh[b0:b0 + BLOCK] = xb
                nb += b
    nb += (T - C) * 2 * K.shape[1] * P["window_bits"] / 8 + (
        (T - C) * 2 * (K.shape[1] / 128) * 4 if P["window_bits"] < 16 else 0)
    if P["window_bits"] < 16:
        Kh[C:T] = quant_rows(K[C:T], P["window_bits"], 128)
        Vh[C:T] = quant_rows(V[C:T], P["window_bits"], 128)
    return Kh, Vh, nb, C


def build_store(K, V, T, P, qstat, attn_w):
    if P.get("kivi"):
        return build_kivi(K, V, T, P["kivi"])
    if P.get("hK"):
        return build_hybrid(K, V, T, P)
    W = P["window"]
    C = max(0, ((T - W) // BLOCK) * BLOCK)
    Kh, Vh = K.clone(), V.clone()
    nbytes = 0.0
    budgets = None
    if P.get("res_adapt") and C > 0:
        # B3: the same total residual budget, shared out in proportion to each
        # block's leftover error energy (floor 16, cap a full block).
        en = [block_error_energy(K[b0:b0 + BLOCK], V[b0:b0 + BLOCK], P)
              for b0 in range(0, C, BLOCK)]
        tot_R = P["resK"] * len(en)
        s = sum(en) or 1.0
        budgets = [int(min(BLOCK - 1, max(16, round(tot_R * e / s)))) for e in en]
    for bi, b0 in enumerate(range(0, C, BLOCK)):
        Pb = dict(P, _R=budgets[bi]) if budgets else P
        kb, vb, nb = compress_block(K[b0:b0 + BLOCK], V[b0:b0 + BLOCK], Pb, qstat,
                                    attn_w[b0:b0 + BLOCK] if attn_w is not None else None)
        Kh[b0:b0 + BLOCK], Vh[b0:b0 + BLOCK] = kb, vb
        nbytes += nb
    hd_all = K.shape[1]
    win = T - C
    if P["window_bits"] < 16:
        Kh[C:T] = quant_rows(K[C:T], P["window_bits"], 128)
        Vh[C:T] = quant_rows(V[C:T], P["window_bits"], 128)
        nbytes += win * 2 * hd_all * P["window_bits"] / 8 + win * 2 * (hd_all / 128) * 4
    else:
        nbytes += win * 2 * hd_all * 2
    return Kh, Vh, nbytes, C


# ────────────────────────────────────────────────────────── evaluate ──
def evaluate(rec, layers, variants, device):
    T = rec["T"]
    out = {}
    for li in layers:
        L = rec["layers"][li]
        if L["q_dec"] is None or L["q_dec"].shape[0] == 0:
            continue
        K = L["k"].float().to(device)
        V = L["v"].float().to(device)
        Qd = L["q_dec"].float().to(device)
        S = Qd.shape[0]
        Ptot = T + S
        K, V = K[:Ptot], V[:Ptot]
        cos, sin, _, D, scale = rope_tables(rec, Ptot, device)
        Hkv, Hq = K.shape[1] // D, Qd.shape[1] // D
        grp = Hq // Hkv
        qpos = torch.arange(T, T + S, device=device)
        Qr = rot(Qd.view(S, Hq, D), cos[qpos], sin[qpos])                   # [S,Hq,D]
        mask = torch.arange(Ptot, device=device)[None, :] > qpos[:, None]   # [S,P]

        def attend(Kx, Vx):
            Kr = rot(Kx.view(Ptot, Hkv, D), cos, sin)
            Kr = Kr.repeat_interleave(grp, 1)                               # [P,Hq,D]
            lg = torch.einsum("shd,phd->hsp", Qr, Kr) * scale
            lg = lg.masked_fill(mask[None], float("-inf"))
            p = torch.softmax(lg, -1)
            o = torch.einsum("hsp,phd->hsd", p, Vx.view(Ptot, Hkv, D).repeat_interleave(grp, 1))
            return p, o

        p0, o0 = attend(K, V)
        # query statistics from PREFILL queries only (no decode information):
        # per kv head, per RoPE pair, mean squared magnitude
        Qp = L["q_pre"].float().to(device).view(-1, Hq, D)
        qe = Qp[..., :D // 2] ** 2 + Qp[..., D // 2:] ** 2                    # [n,Hq,D/2]
        qstat = qe.mean(0).view(Hkv, grp, D // 2).mean(1) * scale ** 2      # [Hkv,D/2]
        # Attention RECEIVED by each prompt token, two sources:
        #   tail -- the last 1024 prompt positions (holds the question here, so
        #           it is the question-aware reading; the runtime's existing
        #           residual fallback already uses the prompt tail)
        #   all  -- queries sampled uniformly over the whole prompt, each key's
        #           mass averaged over the queries that can see it (H2O-style,
        #           query-agnostic)
        qpp = L["q_pre_pos"].to(device)
        Kr_T = rot(K[:T].view(T, Hkv, D), cos[:T], sin[:T]).repeat_interleave(grp, 1)
        ar = torch.arange(T, device=device)

        def received(sel):
            tot = torch.zeros(T, device=device)
            cnt = torch.zeros(T, device=device)
            idx = sel.nonzero().flatten()
            for c0 in range(0, idx.numel(), 64):
                ii = idx[c0:c0 + 64]
                Qt = rot(Qp[ii], cos[qpp[ii]], sin[qpp[ii]])
                lg = torch.einsum("shd,phd->hsp", Qt, Kr_T) * scale
                vis = ar[None, :] <= qpp[ii][:, None]                         # [s,T]
                lg = lg.masked_fill(~vis[None], float("-inf"))
                tot += torch.softmax(lg, -1).mean(0).sum(0)
                cnt += vis.float().sum(0)
            return tot / cnt.clamp(min=1)

        g = torch.Generator(device="cpu").manual_seed(0)
        pick = torch.randperm(qpp.numel(), generator=g)[:256].to(device)
        res256 = torch.zeros_like(qpp, dtype=torch.bool)
        res256[pick] = True
        def received_blockwise(sel):
            # Normalized WITHIN each 1024-token block: what a compressor that
            # sees one block at a time (streaming) can compute.
            idx = sel.nonzero().flatten()
            Qt = rot(Qp[idx], cos[qpp[idx]], sin[qpp[idx]])
            w = torch.zeros(T, device=device)
            for b0 in range(0, T, BLOCK):
                b1 = min(T, b0 + BLOCK)
                lg = torch.einsum("shd,phd->hsp", Qt, Kr_T[b0:b1]) * scale
                vis = ar[None, b0:b1] <= qpp[idx][:, None]
                lg = lg.masked_fill(~vis[None], float("-inf"))
                rows = vis.any(1)
                if not rows.any():
                    continue
                p = torch.softmax(lg[:, rows], -1).mean(0)                  # [s', n]
                w[b0:b1] = p.sum(0) / vis[rows].float().sum(0).clamp(min=1)
            return w

        attn_src = {"tail": received(qpp >= max(0, T - 1024)), "all": received(qpp >= 0),
                    "res256": received(res256), "res256_blk": received_blockwise(res256)}
        for name, P in variants.items():
            Kh, Vh, nb, C = build_store(K, V, T, P, qstat, attn_src[P.get("attn_src", "tail")])
            p1, o1 = attend(Kh, Vh)
            err = ((o1 - o0).norm(dim=-1) / o0.norm(dim=-1).clamp(min=1e-8)).mean().item()
            kl = (p0 * (p0.clamp(min=1e-30).log() - p1.clamp(min=1e-30).log())).sum(-1).mean().item()
            dense_b = T * 2 * K.shape[1] * 2
            out.setdefault(name, []).append({"layer": li, "err": err, "kl": kl,
                                             "bpt": nb / T, "cmp": dense_b / nb,
                                             "compressed_frac": C / T})
    return out


# ─────────────────────────────────────────────────────────── routing ──
def residual_rows(K, V, T, P):
    """Which prompt rows the store keeps as residuals (the router scores a
    block by its anchor and these rows), mirroring compress_block's choice."""
    W = P["window"]
    C = max(0, ((T - W) // BLOCK) * BLOCK)
    mask = torch.zeros(T, dtype=torch.bool, device=K.device)
    for b0 in range(0, C, BLOCK):
        Kb, Vb = K[b0:b0 + BLOCK], V[b0:b0 + BLOCK]
        aK, aV = Kb[:1], Vb[:1]
        dK, dV = Kb[1:] - aK, Vb[1:] - aV
        g = float((dK.pow(2).sum() / dV.pow(2).sum().clamp(min=1e-12)).sqrt().clamp(1, 1e4))
        U, Vh = factor(torch.cat([dK, dV * g], 1), P["rank"], False)
        hd = K.shape[1]
        rec = U @ Vh
        eK, eV = dK - rec[:, :hd], dV - rec[:, hd:] / g
        s = (eK.pow(2).sum(1) + (g * eV).pow(2).sum(1)).sqrt()
        i = torch.topk(s, min(P["resK"], s.numel())).indices
        mask[b0 + 1 + i] = True
    return mask, C


def evaluate_routing(rec, layers, schemes, device, P=None):
    """Attention through the DKV store with each routing scheme, against exact
    dense attention. Returns {scheme: [ {err, kl, mass, blocks} per layer ]}."""
    P = P or dict(BASE, score="joint")
    T = rec["T"]
    out = {}
    for li in layers:
        L = rec["layers"][li]
        if L["q_dec"] is None or L["q_dec"].shape[0] == 0:
            continue
        K = L["k"].float().to(device)
        V = L["v"].float().to(device)
        Qd = L["q_dec"].float().to(device)
        S = Qd.shape[0]
        Ptot = T + S
        K, V = K[:Ptot], V[:Ptot]
        cos, sin, _, D, scale = rope_tables(rec, Ptot, device)
        Hkv, Hq = K.shape[1] // D, Qd.shape[1] // D
        grp = Hq // Hkv
        qpos = torch.arange(T, T + S, device=device)
        Qr = rot(Qd.view(S, Hq, D), cos[qpos], sin[qpos])
        causal = torch.arange(Ptot, device=device)[None, :] > qpos[:, None]   # [S,P]
        Kh, Vh, _nb, C = build_store(K, V, T, P, None, None)
        resm, _ = residual_rows(K, V, T, P)
        nB = C // BLOCK

        def attend(Kx, Vx, extra_mask=None):
            Kr = rot(Kx.view(Ptot, Hkv, D), cos, sin).repeat_interleave(grp, 1)
            lg = torch.einsum("shd,phd->hsp", Qr, Kr) * scale
            m = causal[None].expand(Hq, -1, -1)
            if extra_mask is not None:
                m = m | extra_mask
            lg = lg.masked_fill(m, float("-inf"))
            p = torch.softmax(lg, -1)
            return p, torch.einsum("hsp,phd->hsd", p, Vx.view(Ptot, Hkv, D).repeat_interleave(grp, 1))

        p0, o0 = attend(K, V)                                  # exact dense
        if nB == 0:
            continue
        # per-block quantities on the STORE's keys
        Kr_h = rot(Kh.view(Ptot, Hkv, D), cos, sin).repeat_interleave(grp, 1)   # [P,Hq,D]
        lg_all = torch.einsum("shd,phd->hsp", Qr, Kr_h) * scale                  # [Hq,S,P]
        blk = torch.arange(C, device=device) // BLOCK
        scorable = torch.zeros(C, dtype=torch.bool, device=device)
        scorable[torch.arange(nB, device=device) * BLOCK] = True              # anchors
        scorable |= resm[:C]
        lg_s = lg_all[:, :, :C].masked_fill(~scorable[None, None], float("-inf"))
        # router score per (head, step, block): max over the block's scorable keys
        sc_h = torch.full((Hq, S, nB), float("-inf"), device=device)
        sc_h = sc_h.scatter_reduce(2, blk.view(1, 1, -1).expand(Hq, S, -1), lg_s,
                                   reduce="amax", include_self=True)
        sc = sc_h.max(0).values                                               # [S,nB]
        # landmarks: 16 sub-chunk means of the store's keys per block
        sub = BLOCK // 16
        Kr_c = Kr_h[:C].view(nB * 16, sub, Hq, D).mean(1)                    # [nB*16,Hq,D]
        lg_lm = torch.einsum("shd,phd->hsp", Qr, Kr_c) * scale                # [Hq,S,nB*16]
        sc_lm = torch.maximum(sc_h, lg_lm.view(Hq, S, nB, 16).amax(-1)).max(0).values
        # true attention mass per block (oracle)
        mass_b = torch.zeros(Hq, S, nB, device=device).scatter_add(
            2, blk.view(1, 1, -1).expand(Hq, S, -1), p0[:, :, :C])
        for name, (kind, k) in schemes.items():
            if kind == "all":
                sel = torch.ones(S, nB, dtype=torch.bool, device=device)
                sel_h = None
            elif kind in ("topk", "frac", "landmark", "oracle"):
                kk = k if kind != "frac" else max(16, math.ceil(k * nB))
                kk = min(kk, nB)
                base = {"topk": sc, "frac": sc, "landmark": sc_lm,
                        "oracle": mass_b.mean(0)}[kind]
                sel = torch.zeros(S, nB, dtype=torch.bool, device=device)
                sel.scatter_(1, torch.topk(base, kk, dim=1).indices, True)
                sel_h = None
            elif kind == "topp":
                pr = torch.softmax(sc, -1)
                srt, idx = pr.sort(-1, descending=True)
                keep = (srt.cumsum(-1) - srt) < k
                keep[:, :min(4, nB)] = True
                sel = torch.zeros(S, nB, dtype=torch.bool, device=device)
                sel.scatter_(1, idx, keep)
                sel_h = None
            elif kind == "perhead":
                kk = min(k, nB)
                sel_h = torch.zeros(Hq, S, nB, dtype=torch.bool, device=device)
                sel_h.scatter_(2, torch.topk(sc_h, kk, dim=2).indices, True)
            else:
                raise ValueError(kind)
            if sel_h is None:
                drop = ~sel[:, blk]                                           # [S,C]
                dm = torch.zeros(Hq, S, Ptot, dtype=torch.bool, device=device)
                dm[:, :, :C] = drop[None]
                nsel = float(sel.float().sum(1).mean())
            else:
                dm = torch.zeros(Hq, S, Ptot, dtype=torch.bool, device=device)
                dm[:, :, :C] = ~sel_h[:, :, blk]
                nsel = float(sel_h.float().sum(2).mean())
            p1, o1 = attend(Kh, Vh, dm)
            err = ((o1 - o0).norm(dim=-1) / o0.norm(dim=-1).clamp(min=1e-8)).mean().item()
            kl = (p0 * (p0.clamp(min=1e-30).log() - p1.clamp(min=1e-30).log())).sum(-1)
            kl = kl[torch.isfinite(kl)].mean().item()
            kept = (p0 * (~dm).float()).sum(-1).mean().item()
            out.setdefault(name, []).append({"layer": li, "err": err, "kl": kl,
                                             "mass": kept, "blocks": nsel, "nB": nB})
    return out


ROUTING = {
    "all": ("all", 0), "top16": ("topk", 16), "top32": ("topk", 32),
    "frac25": ("frac", 0.25), "frac50": ("frac", 0.5),
    "topp90": ("topp", 0.90), "topp99": ("topp", 0.99),
    "perhead16": ("perhead", 16), "landmark16": ("landmark", 16),
    "oracle16": ("oracle", 16),
}


BASE = dict(rank=32, split=None, vgain=True, chnorm=False, u_bits=8, u_colscale=False,
            resK=128, resV=128, tiers=True, res_bits=8, score="rel", window=1536,
            window_bits=16)


def V_(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


VARIANTS = {
    # calibration against the tier-3 diagnostic batch
    "base": V_(),
    "notiers": V_(tiers=False),
    "res256": V_(resK=256, resV=256),
    "res512": V_(resK=512, resV=512),
    "rank64": V_(rank=64),
    "resq16": V_(res_bits=16),
    "vgain0": V_(vgain=False),
    # A1: separate K and V factors at the same bytes as joint rank 32
    "A1_16+16": V_(split=(16, 16)),
    "A1_8+24": V_(split=(8, 24)),
    "A1_12+20": V_(split=(12, 20)),
    # A3a: per-column U scales before int8 (free precision)
    "A3_ucol": V_(u_colscale=True),
    # A3b: 4-bit U (per column) and an 8-bit window, saved bytes into residuals
    "A3_u4w8": V_(u_bits=4, u_colscale=True, window_bits=8),
    "A3_u4w8+res": V_(u_bits=4, u_colscale=True, window_bits=8, resK=192, resV=192),
    # A4: outlier-channel normalization before the SVD
    "A4_chnorm": V_(chnorm=True),
    # A2: residuals chosen by importance instead of relative error
    "A2_qenergy": V_(score="qenergy"),
    "A2_attnw": V_(score="attnw"),
    "A2_qe_notier": V_(score="qenergy", tiers=False),
    # round 2: query-agnostic importance, and combinations
    "A2_attnw_all": V_(score="attnw", attn_src="all"),
    "A2_attnwV_all": V_(score="attnwV", attn_src="all"),
    "A2_attnwV_tail": V_(score="attnwV"),
    "vg0+attall": V_(vgain=False, score="attnw", attn_src="all"),
    "u4w8+r48": V_(u_bits=4, u_colscale=True, window_bits=8, rank=48),
    "u4w8+r64": V_(u_bits=4, u_colscale=True, window_bits=8, rank=64),
    "u4w8+r48+vg0+att": V_(u_bits=4, u_colscale=True, window_bits=8, rank=48, vgain=False,
                           score="attnw", attn_src="all"),
    "u4w8+r64+vg0+att": V_(u_bits=4, u_colscale=True, window_bits=8, rank=64, vgain=False,
                           score="attnw", attn_src="all"),
    "u4w8+r64+vg0+att256": V_(u_bits=4, u_colscale=True, window_bits=8, rank=64,
                              vgain=False, score="attnw", attn_src="res256"),
    # runtime-faithful residual rule (joint absolute error, one index set)
    "rt_base": V_(score="joint"),
    "rt_att": V_(score="jointattn", attn_src="res256"),
    "rt_vg0": V_(score="joint", vgain=False),
    "rt_r64": V_(score="joint", rank=64),
    "rt_u4w8r64": V_(score="joint", u_bits=4, u_colscale=True, window_bits=8, rank=64),
    "rt_u4w8r64vg0": V_(score="joint", u_bits=4, u_colscale=True, window_bits=8, rank=64,
                        vgain=False),
    "rt_u4w8r64vg0att": V_(score="jointattn", attn_src="res256", u_bits=4, u_colscale=True,
                           window_bits=8, rank=64, vgain=False),
    "rt_att_blk": V_(score="jointattn", attn_src="res256_blk"),
    "rt_u4w8r64vg0att_blk": V_(score="jointattn", attn_src="res256_blk", u_bits=4,
                               u_colscale=True, window_bits=8, rank=64, vgain=False),
    # ── batch B ──
    "B1_t2": V_(score="joint", tok_bits=2),
    "B1_t2g128": V_(score="joint", tok_bits=2, tok_group=128),
    "B1_t3": V_(score="joint", tok_bits=3),
    "B1_t4": V_(score="joint", tok_bits=4),
    "B1_t2_R32": V_(score="joint", tok_bits=2, resK=32, resV=32),
    "B1_t2_R0": V_(score="joint", tok_bits=2, resK=0, resV=0),
    "B1_t2_u4w8": V_(score="joint", tok_bits=2, u_bits=4, u_colscale=True, window_bits=8),
    "B1_t2_u4w8_R32": V_(score="joint", tok_bits=2, u_bits=4, u_colscale=True,
                         window_bits=8, resK=32, resV=32),
    "B1_t2_u4w8r64vg0": V_(score="joint", tok_bits=2, u_bits=4, u_colscale=True,
                           window_bits=8, rank=64, vgain=False),
    "B1_t2_r16_u4w8_R32": V_(score="joint", tok_bits=2, u_bits=4, u_colscale=True,
                             window_bits=8, rank=16, resK=32, resV=32),
    "B3_adapt": V_(score="joint", res_adapt=True),
    "B3_adapt_u4w8r64vg0": V_(score="joint", res_adapt=True, u_bits=4, u_colscale=True,
                              window_bits=8, rank=64, vgain=False),
    # hybrids: which side wants low rank, which wants quantization
    "H_kq4_vlr32": V_(hK="kq4", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_kq3_vlr32": V_(hK="kq3", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_kq2_vlr32": V_(hK="kq2", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_kq4_vlr16": V_(hK="kq4", hV="lr", hr=16, hR=32, u_bits=4, window_bits=8),
    "H_klr32_vq4": V_(hK="lr", hV="kq4", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_klr32t2_vq2": V_(hK="lr", hV="kq2", hr=32, hR=64, u_bits=4, window_bits=8, htok=2),
    "H_kq4_vq4_w8": V_(hK="kq4", hV="kq4", hr=32, hR=0, u_bits=4, window_bits=8),
    "H_kt4g32_vlr32": V_(hK="kt4g32", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_kt4g64_vlr32": V_(hK="kt4g64", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_kt4g128_vlr32": V_(hK="kt4g128", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "H_kt8g64_vlr32": V_(hK="kt8g64", hV="lr", hr=32, hR=64, u_bits=4, window_bits=8),
    "kivi4": V_(kivi=4),
    "kivi2": V_(kivi=2),
    "u4w8+r40+res160+vg0+att": V_(u_bits=4, u_colscale=True, window_bits=8, rank=40,
                                  resK=160, resV=160, vgain=False, score="attnw",
                                  attn_src="all"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "paper", "results", "diag", "tier1"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--extra", default="", help="JSON dict name -> overrides of BASE")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--routing", action="store_true",
                    help="screen routing schemes on the DKV store instead of store variants")
    args = ap.parse_args()
    if args.routing:
        agg = {}
        for f in sorted(glob.glob(os.path.join(args.dir, "*.pt"))):
            rec = torch.load(f)
            with torch.no_grad():
                res = evaluate_routing(rec, args.layers or sorted(rec["layers"]), ROUTING,
                                       args.device)
            for k, rows in res.items():
                agg.setdefault(k, []).extend(dict(r, prompt=rec["key"], T=rec["T"])
                                             for r in rows)
            print("  %s done" % rec["key"], flush=True)
        for T_lo, T_hi, tag in ((0, 48000, "32k"), (48000, 10 ** 9, "64k")):
            print("\n[%s]  %-11s %8s %9s %8s %7s" % (tag, "scheme", "out_err", "attn_KL",
                                                     "mass", "blocks"))
            for k, rows in agg.items():
                rr = [r for r in rows if T_lo <= r["T"] < T_hi]
                if not rr:
                    continue
                m = lambda key: sum(r[key] for r in rr) / len(rr)       # noqa: E731
                print("       %-11s %8.4f %9.4f %8.4f %5.1f/%d" % (
                    k, m("err"), m("kl"), m("mass"), m("blocks"), rr[0]["nB"]))
        return
    variants = {k: VARIANTS[k] for k in args.variants if k in VARIANTS}
    if args.extra:
        for k, o in json.loads(args.extra).items():
            variants[k] = V_(**o)
    agg = {}
    files = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    if not files:
        raise SystemExit("no captures in " + args.dir)
    for f in files:
        rec = torch.load(f)
        layers = args.layers or sorted(rec["layers"])
        with torch.no_grad():
            res = evaluate(rec, layers, variants, args.device)
        for k, rows in res.items():
            agg.setdefault(k, []).extend(dict(r, prompt=rec["key"]) for r in rows)
        print("  %s done" % rec["key"], flush=True)
    b = agg["base"] if "base" in agg else None
    print("\n%-14s %9s %9s %9s %9s %7s" % ("variant", "out_err", "vs base", "attn_KL",
                                          "B/tok", "cmp"))
    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)          # noqa: E731
    for k, rows in agg.items():
        rel = mean(rows, "err") / mean(b, "err") if b else float("nan")
        print("%-14s %9.4f %8.2fx %9.5f %9.0f %7.2f" % (
            k, mean(rows, "err"), rel, mean(rows, "kl"), mean(rows, "bpt"), mean(rows, "cmp")))
    with open(os.path.join(args.dir, "screen.jsonl"), "a") as fh:
        for k, rows in agg.items():
            fh.write(json.dumps({"variant": k, "params": variants.get(k), "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
