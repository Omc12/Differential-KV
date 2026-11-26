#!/usr/bin/env python3
"""The runtime router's own selections, scored against exact attention.

WHY. Offline on exact keys (diag_route_oracle.py) normalising each head before
the cross-head max cuts Qwen2.5-7B's routed attention-output error 0.066 ->
0.042, yet real decode got worse (KL 0.101 -> 0.217). The runtime router does
not see exact keys: it sees each block's anchor plus <=64 residual rows read
back from the codes. This runs real DKV decode, and at every layer's first
routing call (one decode step) scores, against EXACT keys and values captured
from the projections during prefill:

  raw / vote         the runtime router, DKV_ROUTE_HEADS=max / vote
  sig_raw / sig_hn   the same rules on EXACT keys at the router's own rows
                     (anchor + the residual rows it reads): position vs codes
  all_raw / all_hn   the same rules on exact keys over every row (the ceiling)
  oracle             top-K by mean per-head mass

Score: attention-output rel err over the candidate blocks plus every token
after the last block (the dense window), all heads concatenated; and the mean
per-head attention mass the selection drops.

USAGE
  python benchmarks/diag_route_runtime.py --model Qwen/Qwen2.5-7B-Instruct \
      --ruler 32768/niah_multivalue#0
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE = os.path.join(ROOT, "ACTIVE_RUNTIME")
BENCH = os.path.join(ROOT, "benchmarks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ruler", required=True, help="LEN/task#idx")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    os.environ.setdefault("DKV_STORE", "codes")
    os.environ.setdefault("DKV_ROUTE_PROBE", "0")
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    import torch
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from run_ruler_cuda import build_ruler_prompt
    from runtime.dkv_attention import resolve_decoder_layers
    import native_core.srl.query_router as QR

    w = DKVHFWrapper(model_id=args.model, config={"quantization": "nf4", "preset": "mid"})
    w.ensure_loaded()
    tok, model = w.tokenizer, w.model
    layers = resolve_decoder_layers(model)

    exact = {}
    hooks = []
    for li, l in enumerate(layers):
        at = getattr(l, "self_attn", None)
        if at is None or getattr(at, "k_proj", None) is None:
            continue
        ksrc = getattr(at, "k_norm", None) or at.k_proj

        def hk(_m, _i, out, li=li):
            if out.shape[-2 if out.dim() == 4 else 1] > 1:
                exact.setdefault(li, {"k": [], "v": []})["k"].append(
                    out[0].reshape(out.shape[1], -1).detach().half())

        def hv(_m, _i, out, li=li):
            if out.shape[1] > 1:
                exact.setdefault(li, {"k": [], "v": []})["v"].append(out[0].detach().half())
        hooks += [ksrc.register_forward_hook(hk), at.v_proj.register_forward_hook(hv)]

    cur = {"layer": -1}
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_pre_hook(
            lambda _m, _a, i=i: cur.__setitem__("layer", i)))

    rows, done = {}, set()
    orig = QR.route_blocks_relevance

    def route(Q, pool, block_indices, anchor_indices, scale, cos=None, sin=None,
              srl_state=None, layer_idx=None):
        li = cur["layer"]
        out = orig(Q, pool, block_indices, anchor_indices, scale, cos, sin,
                   srl_state, layer_idx)
        if li in done or li not in exact or Q.dim() != 2 or cos is None:
            return out
        done.add(li)
        try:
            with torch.no_grad():
                rows[li] = _score(QR, orig, Q, pool, block_indices, anchor_indices,
                                  scale, cos, sin, exact[li], out)
        except Exception as e:                                   # noqa: BLE001
            print(f"[diag] layer {li} failed: {e}", flush=True)
        return out

    QR.route_blocks_relevance = route

    L, rest = args.ruler.split("/", 1)
    task, idx = rest.split("#")
    data = args.data or os.path.join(ROOT, "paper", "results", "ruler_data",
                                      args.model.split("/")[-1])
    with open(os.path.join(data, L, task, "validation.jsonl"), encoding="utf-8") as f:
        item = json.loads([x for x in f if x.strip()][int(idx)])
    w.active_session = "route-diag"
    text = w.generate(build_ruler_prompt(tok, item), max_new_tokens=3, temperature=0.0,
                      top_p=1.0, repetition_penalty=1.0)
    for h in hooks:
        h.remove()
    print(f"generated {text!r}  gold {item['outputs']}")
    names = ("raw", "vote", "sig_vote", "dev_raw", "dev_vote", "all_raw", "all_vote",
             "oracle")
    print("attention-output rel err | missed mass (mean over heads)")
    print(f"{'layer':>5} {'N':>3} {'K':>3} " + " ".join(f"{n:>14}" for n in names))
    for li in sorted(rows):
        r = rows[li]
        print(f"{li:>5} {r['N']:>3} {r['K']:>3} " + " ".join(
            f"{r[n][0]:>7.4f}|{r[n][1]:<6.3f}" for n in names))
    if rows:
        print(f"{'mean':>13} " + " ".join(
            f"{sum(rows[l][n][0] for l in rows) / len(rows):>7.4f}|"
            f"{sum(rows[l][n][1] for l in rows) / len(rows):<6.3f}" for n in names))


def _score(QR, orig, Q, pool, block_indices, anchor_indices, scale, cos, sin, ex, sel_now):
    import torch
    dev = Q.device
    K = torch.cat(ex["k"]).to(dev).float()                   # [P, Hkv*D] pre-RoPE
    V = torch.cat(ex["v"]).to(dev).float()
    H, D = Q.shape
    Hkv = K.shape[1] // D
    P = K.shape[0]
    rot = cos.shape[-1]
    cf, sf = cos.reshape(-1, rot).float(), sin.reshape(-1, rot).float()
    pos = torch.arange(P, device=dev).clamp(max=cf.shape[0] - 1)
    Kr = QR._rope_partial(K.view(P, Hkv, D), cf[pos].unsqueeze(1), sf[pos].unsqueeze(1))
    g = H // Hkv
    Kr = Kr.repeat_interleave(g, 1)                          # [P, H, D]
    Vh = V.view(P, Hkv, D).repeat_interleave(g, 1)
    lg = torch.einsum("hd,phd->hp", Q.float(), Kr) * scale   # [H, P]

    anc = anchor_indices.long().to(dev)
    N = anc.numel()
    order = torch.argsort(anc)
    a_sorted = anc[order]
    span = int((a_sorted[1:] - a_sorted[:-1]).max()) if N > 1 else P
    ends = torch.cat([a_sorted[1:], (a_sorted[-1:] + span).clamp(max=P)])
    end_of = torch.empty_like(anc)
    end_of[order] = ends
    tail0 = int(ends.max())
    blk = torch.full((P,), -1, dtype=torch.long, device=dev)
    for n in range(N):
        blk[int(anc[n]):int(end_of[n])] = n
    window = torch.arange(P, device=dev) >= tail0
    in_cand = (blk >= 0) | window
    lgc = lg.masked_fill(~in_cand, float("-inf"))
    p = torch.softmax(lgc, -1)
    o_full = torch.einsum("hp,phd->hd", p, Vh)
    mass = torch.zeros(H, N, device=dev).index_add_(1, blk.clamp(min=0), p * (blk >= 0))

    def evaluate(sel_pos):
        keep = torch.zeros(N, dtype=torch.bool, device=dev)
        keep[sel_pos] = True
        tok_keep = window | ((blk >= 0) & keep[blk.clamp(min=0)])
        o = torch.einsum("hp,phd->hd", torch.softmax(lg.masked_fill(~tok_keep, float("-inf")), -1), Vh)
        return (float((o - o_full).norm() / o_full.norm()),
                float(mass[:, ~keep].sum(1).mean()))

    slots = block_indices.long()
    slot_to_pos = {int(s): i for i, s in enumerate(slots.tolist())}

    def as_pos(sel_slots):
        return torch.tensor([slot_to_pos[int(s)] for s in sel_slots.tolist()], device=dev)

    out = {"N": N, "K": int(sel_now.numel())}
    saved = QR._ROUTE_HEADS
    for name, flag in (("raw", "max"), ("vote", "vote")):
        QR._ROUTE_HEADS = flag
        out[name] = evaluate(as_pos(orig(Q, pool, block_indices, anchor_indices, scale, cos, sin)))
    QR._ROUTE_HEADS = saved
    k = out["K"]

    def pick(z, norm):                                       # z [H, N] per-head block score
        if norm:
            z = z - torch.logsumexp(z, -1, keepdim=True)
        return torch.topk(z.amax(0), k).indices

    # exact keys at the router's own rows: anchor + first 64 residual rows
    rp = getattr(pool, "residual_K_positions", None)
    zs = torch.full((H, N), float("-inf"), device=dev)
    for n in range(N):
        ps = [int(anc[n])]
        if rp is not None:
            ps += [int(anc[n]) + 1 + int(x) for x in rp[slots[n], :64].tolist() if int(x) >= 0]
        ps = [x for x in ps if x < P]
        zs[:, n] = lg[:, ps].amax(1)
    # Alternative row choice at the SAME count: per KV head, the rows whose
    # (pre-RoPE) key deviates most from the block's mean key, round-robin over
    # heads. Keys only, per head: what routing needs, nothing the store needs.
    Kh = K.view(P, Hkv, D)
    zd = torch.full((H, N), float("-inf"), device=dev)
    for n in range(N):
        a, e = int(anc[n]), int(end_of[n])
        n_sig = 0
        if rp is not None:
            n_sig = int((rp[slots[n], :64] >= 0).sum())
        blk_k = Kh[a + 1:e]                                   # [T, Hkv, D]
        ps = [a]
        if n_sig > 0 and blk_k.shape[0] > 0:
            dev_h = (blk_k - blk_k.mean(0, keepdim=True)).norm(dim=-1)   # [T, Hkv]
            ordr = dev_h.argsort(0, descending=True)                     # [T, Hkv]
            seen = set()
            for r_ in range(ordr.shape[0]):
                for h_ in range(Hkv):
                    t_ = int(ordr[r_, h_])
                    if t_ not in seen:
                        seen.add(t_)
                        ps.append(a + 1 + t_)
                    if len(seen) >= n_sig:
                        break
                if len(seen) >= n_sig:
                    break
        ps = [x for x in ps if x < P]
        zd[:, n] = lg[:, ps].amax(1)
    za = torch.full((H, N), float("-inf"), device=dev)
    for n in range(N):
        za[:, n] = lg[:, int(anc[n]):int(end_of[n])].amax(1)

    def vote(z):                                             # each head spreads one vote
        return torch.topk(torch.softmax(z, -1).mean(0), k).indices

    out["dev_vote"] = evaluate(vote(zd))
    out["dev_raw"] = evaluate(pick(zd, False))
    out["sig_vote"] = evaluate(vote(zs))
    out["all_vote"] = evaluate(vote(za))
    out["sig_raw"] = evaluate(pick(zs, False))
    out["sig_hn"] = evaluate(pick(zs, True))
    out["all_raw"] = evaluate(pick(za, False))
    out["all_hn"] = evaluate(pick(za, True))
    out["oracle"] = evaluate(torch.topk(mass.mean(0), k).indices)
    return out


if __name__ == "__main__":
    main()
