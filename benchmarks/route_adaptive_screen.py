#!/usr/bin/env python3
"""Offline screen of block-count rules on captured real KV (tier1 captures).

For each decode query of each captured layer, against exact dense attention,
reports for every rule the mean routed blocks, the true attention mass the
routed set covers (blocks + exact window), and the KL of attending only that
set through the EXACT keys (so routing error is measured alone, not mixed with
store error):

  box<K>        the shipped router: exact post-RoPE key boxes per KV head
                (Quest bound q.mid + |q|.half), max over query heads, top-K
  oracle<K>     top-K by true per-block mass (an upper bound for any K-rule)
  ada<P>        adaptive: box top-16 floor, then add blocks by an estimated
                per-block log-mass until the estimated share of the softmax
                (window included) reaches P, capped at --kmax
  adaq<P>       same, ranked and counted by the estimate only, floor 16
  est<K>        top-K by the estimate alone (a fixed-K router)
  adam<P>       adaq plus the box's top-4 (a needle the moments cannot see)
  adanw<P>      adam without the window term (shares among blocks only)
  adab<P>       box top-16 floor (today's set, unchanged), grown by the
                estimate, no window term -- grow-only

The per-block log-mass estimate is a moment estimate recorded at compression
time with the box: lse_b(q) ~ log n + s q.mu + s^2/2 q^T C q, with C the
block's key covariance kept as its top-c principal directions plus an
isotropic remainder, then clipped to the bounds every block satisfies:
    log n + s q.mu  <=  lse_b  <=  log n + s * max_k q.k  (<= box bound)
The window's lse is exact (the runtime holds its keys).

    python benchmarks/route_adaptive_screen.py --dir paper/results/diag/tier1_qwen
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier1_screen import BLOCK, rope_tables, rot  # noqa: E402


def block_stats(Kr_blocks, c):
    """Kr_blocks [nB, n, Hkv, D] exact post-RoPE keys -> box + moments."""
    mn = Kr_blocks.amin(1)                                   # [nB,Hkv,D]
    mx = Kr_blocks.amax(1)
    mu = Kr_blocks.mean(1)                                   # [nB,Hkv,D]
    if c == -2:
        # Diagonal covariance: q^T diag(var) q, one vector per block, no eig.
        var = Kr_blocks.var(1, unbiased=False)               # [nB,Hkv,D]
        return mn, mx, mu, var, None
    X = (Kr_blocks - mu[:, None]).permute(0, 2, 1, 3)        # [nB,Hkv,n,D]
    n = X.shape[2]
    C = X.transpose(-1, -2) @ X / n                          # [nB,Hkv,D,D]
    ev, evec = torch.linalg.eigh(C)                          # ascending
    D = C.shape[-1]
    c = D if c < 0 else min(c, D)                            # -1 = full covariance
    k = D - c                                                # eigh is ascending
    top = evec[..., k:] * ev[..., k:].clamp(min=0).sqrt().unsqueeze(-2)     # [nB,Hkv,D,c]
    rest = (ev[..., :k].clamp(min=0).sum(-1) / max(1, k))                   # [nB,Hkv]
    return mn, mx, mu, top.transpose(-1, -2), rest           # W [nB,Hkv,c,D]


def estimate_lse(Qh, mu, W, rest, n, scale, ub):
    """Qh [Hq,S,D] (head h reads kv head h//grp); returns [Hq,S,nB]."""
    Hq = Qh.shape[0]
    Hkv = mu.shape[1]
    grp = Hq // Hkv
    kv = torch.arange(Hq, device=Qh.device) // grp
    muh = mu[:, kv]                                          # [nB,Hq,D]
    Wh = W[:, kv]                                            # [nB,Hq,c,D]
    rh = rest[:, kv] if rest is not None else None           # [nB,Hq]
    lin = torch.einsum("hsd,bhd->hsb", Qh, muh) * scale
    if rest is None:                                         # diagonal: W is var
        var = torch.einsum("hsd,bhd->hsb", Qh * Qh, Wh)
    else:
        proj = torch.einsum("hsd,bhcd->hsbc", Qh, Wh)
        q2 = (Qh * Qh).sum(-1, keepdim=True)                 # [Hq,S,1]
        var = (proj ** 2).sum(-1) + rh.T[:, None, :] * q2    # rest: isotropic
    est = math.log(n) + lin + 0.5 * scale * scale * var
    lo = math.log(n) + lin
    return torch.minimum(torch.maximum(est, lo), math.log(n) + ub)


def evaluate(rec, layers, rules, device, c, kmax, window):
    T = rec["T"]
    out = {}
    for li in layers:
        L = rec["layers"][li]
        if L.get("q_dec") is None or L["q_dec"].shape[0] == 0:
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
        Qr = rot(Qd.view(S, Hq, D), cos[qpos], sin[qpos])           # [S,Hq,D]
        Kr = rot(K.view(Ptot, Hkv, D), cos, sin)                    # [P,Hkv,D]
        nB = max(0, (T - window) // BLOCK)
        if nB < 2:
            continue
        C = nB * BLOCK
        Krg = Kr.repeat_interleave(grp, 1)
        lg = torch.einsum("shd,phd->hsp", Qr, Krg) * scale          # [Hq,S,P]
        causal = torch.arange(Ptot, device=device)[None, :] > qpos[:, None]
        lg = lg.masked_fill(causal[None], float("-inf"))
        p0 = torch.softmax(lg, -1)
        o0 = torch.einsum("hsp,phd->hsd", p0, V.view(Ptot, Hkv, D).repeat_interleave(grp, 1))
        blk = torch.arange(C, device=device) // BLOCK
        mass_b = torch.zeros(Hq, S, nB, device=device).scatter_add(
            2, blk.view(1, 1, -1).expand(Hq, S, -1), p0[:, :, :C])
        # exact per-block lse (for the estimate's own accuracy)
        lse_true = torch.logsumexp(lg[:, :, :C].reshape(Hq, S, nB, BLOCK), -1)
        lse_win = torch.logsumexp(lg[:, :, C:], -1)                 # [Hq,S]

        mn, mx, mu, W, rest = block_stats(Kr[:C].view(nB, BLOCK, Hkv, D), c)
        mid, half = (mx + mn) / 2, (mx - mn) / 2
        kv = torch.arange(Hq, device=device) // grp
        Qh = Qr.permute(1, 0, 2)                                    # [Hq,S,D]
        ub = (torch.einsum("hsd,bhd->hsb", Qh, mid[:, kv])
              + torch.einsum("hsd,bhd->hsb", Qh.abs(), half[:, kv])) * scale  # [Hq,S,nB]
        box = ub.amax(0)                                            # [S,nB]
        est = estimate_lse(Qh, mu, W, rest, BLOCK, scale, ub)       # [Hq,S,nB]
        # estimate accuracy: rank correlation of est vs true lse, per (h,s)
        def _rank(x):
            return x.argsort(-1).argsort(-1).float()
        rc = torch.nn.functional.cosine_similarity(
            _rank(est) - (nB - 1) / 2, _rank(lse_true) - (nB - 1) / 2, dim=-1).mean().item()
        # estimated share of each block in the full softmax, averaged over heads
        tot = torch.logsumexp(torch.cat([est, lse_win[..., None]], -1), -1, keepdim=True)
        share = torch.exp(est - tot).mean(0)                        # [S,nB]
        win_share = torch.exp(lse_win - tot[..., 0]).mean(0)       # [S]
        # Without the window (the runtime router does not hold its keys):
        # shares among compressed blocks only.
        share_nw = torch.softmax(est, -1).mean(0)                   # [S,nB]

        def pick(kind, arg):
            sel = torch.zeros(S, nB, dtype=torch.bool, device=device)
            if kind == "box":
                kk = min(arg, nB)
                sel.scatter_(1, torch.topk(box, kk, 1).indices, True)
            elif kind == "est":
                kk = min(arg, nB)
                sel.scatter_(1, torch.topk(share, kk, 1).indices, True)
            elif kind == "oracle":
                kk = min(arg, nB)
                sel.scatter_(1, torch.topk(mass_b.mean(0), kk, 1).indices, True)
            elif kind == "all":
                sel[:] = True
            elif kind in ("ada", "adaq", "adam", "adanw", "adab"):
                P = arg
                if kind in ("adanw", "adab"):
                    sh, ws = share_nw, torch.zeros_like(win_share)
                else:
                    sh, ws = share, win_share
                floor = min(16, nB)
                if kind in ("ada", "adab"):
                    sel.scatter_(1, torch.topk(box, floor, 1).indices, True)
                else:
                    sel.scatter_(1, torch.topk(sh, floor, 1).indices, True)
                if kind in ("adam", "adanw"):
                    sel.scatter_(1, torch.topk(box, min(4, nB), 1).indices, True)
                for s in range(S):
                    cov = ws[s] + sh[s][sel[s]].sum()
                    if cov >= P:
                        continue
                    order = torch.argsort(sh[s], descending=True)
                    for b in order.tolist():
                        if sel[s].sum() >= min(kmax, nB):
                            break
                        if sel[s, b]:
                            continue
                        sel[s, b] = True
                        cov = cov + sh[s, b]
                        if cov >= P:
                            break
            return sel

        for name, (kind, arg) in rules.items():
            sel = pick(kind, arg)
            drop = torch.zeros(Hq, S, Ptot, dtype=torch.bool, device=device)
            drop[:, :, :C] = ~sel[:, blk][None]
            p1 = torch.softmax(lg.masked_fill(drop, float("-inf")), -1)
            o1 = torch.einsum("hsp,phd->hsd", p1, V.view(Ptot, Hkv, D).repeat_interleave(grp, 1))
            err = ((o1 - o0).norm(dim=-1) / o0.norm(dim=-1).clamp(min=1e-8)).mean().item()
            kl = (p0 * (p0.clamp(min=1e-30).log() - p1.clamp(min=1e-30).log())).sum(-1)
            kl = kl[torch.isfinite(kl)].mean().item()
            kept = (p0 * (~drop).float()).sum(-1).mean().item()
            nsel = sel.float().sum(1)
            out.setdefault(name, []).append(dict(
                layer=li, err=err, kl=kl, mass=kept, blocks=nsel.mean().item(),
                bmax=nsel.max().item(), nB=nB, rankcorr=rc))
    return out


RULES = {
    "box16": ("box", 16), "box32": ("box", 32), "box64": ("box", 64),
    "oracle16": ("oracle", 16), "all": ("all", 0),
    "est16": ("est", 16), "est32": ("est", 32),
    "ada90": ("ada", 0.90), "ada95": ("ada", 0.95), "ada98": ("ada", 0.98),
    "adaq90": ("adaq", 0.90), "adaq95": ("adaq", 0.95), "adaq98": ("adaq", 0.98),
    "adam95": ("adam", 0.95), "est24": ("est", 24),
    "adab80": ("adab", 0.80), "adab90": ("adab", 0.90), "adab95": ("adab", 0.95),
    "adanw80": ("adanw", 0.80), "adanw90": ("adanw", 0.90), "adanw95": ("adanw", 0.95),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "paper", "results", "diag", "tier1_qwen"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--c", type=int, default=4, help="principal directions per block (0 = isotropic only, -1 = full, -2 = diagonal)")
    ap.add_argument("--kmax", type=int, default=64)
    ap.add_argument("--window", type=int, default=3584)
    ap.add_argument("--rules", nargs="*", default=list(RULES))
    args = ap.parse_args()
    rules = {k: RULES[k] for k in args.rules}
    agg = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.pt"))):
        rec = torch.load(f, weights_only=False)
        with torch.no_grad():
            res = evaluate(rec, sorted(rec["layers"]), rules, args.device, args.c,
                           args.kmax, args.window)
        for k, rows in res.items():
            agg.setdefault(k, []).extend(dict(r, prompt=rec["key"], T=rec["T"]) for r in rows)
        print("  %s done" % rec["key"], flush=True)
    for T_lo, T_hi, tag in ((0, 48000, "32k"), (48000, 10 ** 9, "64k")):
        print("\n[%s] %-9s %8s %9s %8s %8s %6s %8s" % (
            tag, "rule", "out_err", "attn_KL", "mass", "blocks", "max", "rankcor"))
        for k in rules:
            rr = [r for r in agg.get(k, []) if T_lo <= r["T"] < T_hi]
            if not rr:
                continue
            m = lambda key: sum(r[key] for r in rr) / len(rr)        # noqa: E731
            print("     %-9s %8.4f %9.4f %8.4f %5.1f/%-3d %6.0f %8.3f" % (
                k, m("err"), m("kl"), m("mass"), m("blocks"), rr[0]["nB"],
                max(r["bmax"] for r in rr), m("rankcorr")))


if __name__ == "__main__":
    main()
