#!/usr/bin/env python3
"""How much attention mass does each block-selection RULE drop? Dense, exact.

WHY. Routed decode loses RULER items on Qwen2.5-7B that every-block decode
answers (63.8 vs 88.8 at 32k). This separates the selection RULE from the
store: it runs the dense model, takes the first decode step's real post-RoPE
query and the exact cached keys, cuts the compressed region into the runtime's
blocks, and scores each rule by the softmax mass of the blocks it drops,
per head. The recent window is always kept, as in the runtime.

Rules (all on EXACT keys, so the residual router's best case):
  maxlogit   max over heads of max q.k in the block (the runtime's rule)
  headnorm   max over heads of the block's max logit minus that head's own
             log-normaliser over the max logits it can see (window included)
  hn_nowin   the same, normalised over the compressed blocks only (what a
             router that never sees the window can compute)
  lse        max over heads of the block's per-head log mass share (oracle
             per head, upper bound for any per-head-normalised rule)
  oracle     top-K by mean over heads of the block's mass (best mean)

USAGE
  python benchmarks/diag_route_oracle.py --model Qwen/Qwen2.5-7B-Instruct \
      --ruler 32768/niah_multikey_2#0 [--block 1024 --window 1024 --k 16]
  python benchmarks/diag_route_oracle.py --model ibm-granite/... --ctx 32768
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE = os.path.join(ROOT, "ACTIVE_RUNTIME")
BENCH = os.path.join(ROOT, "benchmarks")


def _prompt(tok, args):
    if args.ruler:
        sys.path.insert(0, BENCH)
        from run_ruler_cuda import build_ruler_prompt
        L, rest = args.ruler.split("/", 1)
        task, idx = rest.split("#")
        data = args.data or os.path.join(ROOT, "paper", "results", "ruler_data",
                                          args.model.split("/")[-1])
        with open(os.path.join(data, L, task, "validation.jsonl"), encoding="utf-8") as f:
            r = json.loads([l for l in f if l.strip()][int(idx)])
        return build_ruler_prompt(tok, r), r["outputs"]
    sys.path.insert(0, BENCH)
    from niah_recall import build_prompt
    return build_prompt(tok, args.ctx, args.depth), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ruler", default=None, help="LEN/task#idx")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"device_map": "cuda", "dtype": torch.bfloat16, "attn_implementation": "sdpa",
          "quantization_config": BitsAndBytesConfig(
              load_in_4bit=True, bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()

    stats = {}
    rules = ("maxlogit", "headnorm", "hn_nowin", "lse", "oracle")

    def analyse(module, q, k, v):
        # q [Hq, D] post-RoPE (decode step), k/v [Hkv, S, D] post-RoPE
        Hq, Hkv = q.shape[0], k.shape[0]
        S = k.shape[1]
        scale = float(getattr(module, "scaling", q.shape[-1] ** -0.5))
        kk = k.repeat_interleave(Hq // Hkv, dim=0).float()
        vv = v.repeat_interleave(Hq // Hkv, dim=0).float()
        lg = torch.einsum("hd,hsd->hs", q.float(), kk) * scale          # [Hq, S]
        o_full = torch.einsum("hs,hsd->hd", torch.softmax(lg, -1), vv)
        comp = max(0, S - args.window)
        nb = comp // args.block
        if nb <= args.k:
            return
        cut = nb * args.block                       # blocks [0, cut), rest always kept
        p = torch.softmax(lg, dim=-1)
        m = p[:, :cut].reshape(Hq, nb, args.block).sum(-1)              # [Hq, nb]
        zb = lg[:, :cut].reshape(Hq, nb, args.block).amax(-1)           # [Hq, nb]
        ztail = lg[:, cut:].amax(-1, keepdim=True)                      # [Hq, 1]
        norm = torch.logsumexp(torch.cat([zb, ztail], dim=1), dim=1, keepdim=True)
        norm_b = torch.logsumexp(zb, dim=1, keepdim=True)              # blocks only
        score = {
            "maxlogit": zb.amax(0),
            "headnorm": (zb - norm).amax(0),
            "hn_nowin": (zb - norm_b).amax(0),
            "lse": torch.log(m.clamp_min(1e-30)).amax(0),
            "oracle": m.mean(0),
        }
        row = {"nb": nb, "logit_sd": lg.std(-1).cpu(),
               "logit_max": lg.amax(-1).cpu()}
        for r in rules:
            sel = torch.topk(score[r], args.k).indices
            keep = torch.zeros(nb, dtype=torch.bool, device=m.device)
            keep[sel] = True
            miss = m[:, ~keep].sum(-1)                                   # [Hq]
            row[r] = (float(miss.mean()), float(miss.max()))
            # attention OUTPUT error of the routed set, whole layer (heads concatenated)
            tok = torch.cat([keep.repeat_interleave(args.block),
                             torch.ones(S - cut, dtype=torch.bool, device=m.device)])
            o = torch.einsum("hs,hsd->hd",
                             torch.softmax(lg.masked_fill(~tok, float("-inf")), -1), vv)
            row["oe_" + r] = float((o - o_full).norm() / o_full.norm())
            if r == "maxlogit":
                # which heads decided the kept blocks
                row["winners"] = len(set(zb[:, sel].argmax(0).tolist()))
        stats[int(module.layer_idx)] = row

    base = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def wrapped(module, query, key, value, attention_mask, **kw2):
        if query.shape[2] == 1 and int(module.layer_idx) not in stats:
            with torch.no_grad():
                analyse(module, query[0, :, 0], key[0], value[0])
        return base(module, query, key, value, attention_mask, **kw2)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped

    prompt, gold = _prompt(tok, args)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    with torch.inference_mode():
        for i in range(0, len(ids), args.chunk):
            ch = ids[i:i + args.chunk]
            out = model(input_ids=torch.tensor([ch], device="cuda"),
                        position_ids=torch.arange(i, i + len(ch), device="cuda").unsqueeze(0),
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
        t1 = int(out.logits[0, -1].argmax())
        model(input_ids=torch.tensor([[t1]], device="cuda"),
              position_ids=torch.tensor([[len(ids)]], device="cuda"),
              past_key_values=cache, use_cache=True)
    if not stats:
        sys.exit("no decode-step attention call was captured")

    print(f"tokens {len(ids)}  block {args.block}  window {args.window}  K {args.k}"
          f"  gold {gold}  first token {tok.decode([t1])!r}")
    print("missed attention mass per head (mean / worst head), lower is better")
    print(f"{'layer':>5} {'nb':>3} {'win':>3} " + " ".join(f"{r:>15}" for r in rules)
          + "   logit sd min/max   logit max min/max")
    agg = {r: [] for r in rules}
    for li in sorted(stats):
        s = stats[li]
        for r in rules:
            agg[r].append(s[r])
        sd, mx = s["logit_sd"], s["logit_max"]
        print(f"{li:>5} {s['nb']:>3} {s['winners']:>3} "
              + " ".join(f"{s[r][0]:>7.3f}/{s[r][1]:<7.3f}" for r in rules)
              + f"   {float(sd.min()):6.2f}/{float(sd.max()):<6.2f}"
              f"   {float(mx.min()):7.2f}/{float(mx.max()):<7.2f}")
    print(f"{'mean':>13} " + " ".join(
        f"{sum(a for a, _ in agg[r]) / len(agg[r]):>7.3f}/{max(b for _, b in agg[r]):<7.3f}"
        for r in rules))
    print("\nattention-output rel err of the routed set (all heads), lower is better")
    print(f"{'layer':>5} " + " ".join(f"{r:>9}" for r in rules))
    for li in sorted(stats):
        print(f"{li:>5} " + " ".join(f"{stats[li]['oe_' + r]:>9.4f}" for r in rules))
    print(f"{'mean':>5} " + " ".join(
        f"{sum(stats[li]['oe_' + r] for li in stats) / len(stats):>9.4f}" for r in rules))


if __name__ == "__main__":
    main()
