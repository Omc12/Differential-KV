#!/usr/bin/env python3
"""How far does a KV-compression setting move the model's actual beliefs?

WHY THIS EXISTS
---------------
Needle recall could not separate the arms on this hardware: at the contexts
where DKV compresses, the small models here fail the needle task even under a
DENSE control, and at the contexts where they pass, DKV does not compress
(colab/srl_tradeoffs.py records the attempts). Generated text is no better --
at 32k every arm emits degenerate output, so comparing completions compares
garbage to garbage.

But a quality question does not need the model to SUCCEED at a task. It only
needs the model's next-token distribution. This measures, against the dense
control on an identical prefix:

    top-1 agreement   does the arm pick the same next token as dense?
    KL(dense || arm)  how much probability mass moved?
    dense-top1 rank   where did dense's choice land in the arm's ranking?

FIRST DECODE STEP ONLY, and that is deliberate. Greedy decode diverges: once
two arms pick different tokens they are conditioning on different prefixes and
every later comparison is measuring the divergence rather than the compression.
The first step is the only one where the prefix is guaranteed identical, so it
is the only honest single-forward comparison. n comes from prompts, not steps.

THE CONTROL THAT MATTERS is `baseline` -- DKV with the feature off. Any arm has
to be read against how far DKV ALREADY sits from dense, because that gap is the
price of compression itself and is not the feature's fault. An arm whose KL is
indistinguishable from baseline's has added nothing measurable.

USAGE
    python colab/logit_fidelity.py --arms dense baseline basis0.50 basis0.25
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")

ARMS = {
    "dense":      {"DKV_COMPRESSED_DECODE": "0"},
    "baseline":   {},
    "basis0.50":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.50"},
    "basis0.25":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25"},
    "basis0.125": {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.125"},
}
_ARM_KEYS = ("DKV_SHARED_BASIS", "DKV_SHARED_BASIS_FRAC", "DKV_COMPRESSED_DECODE")


def run_one_arm(args):
    os.environ.pop("DKV_SHARED_BASIS", None)
    for k in _ARM_KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[args.arm])
    os.environ.setdefault("DKV_RSVD_SEED", "1234")
    os.environ.setdefault("DKV_SVD_SEED", "1234")
    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")
    os.environ.setdefault("DKV_ROUTE_PROBE", "0")

    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    import torch
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt

    w = DKVHFWrapper(model_id=args.model,
                     config={"quantization": None, "rank": 32, "block_size": 256,
                             "micro_block_size": 256, "preset": "mid"})
    w.ensure_loaded()
    tok, model = w.tokenizer, w.model

    # Capture the lm_head output. generate() runs chunked prefill then decode,
    # so several calls land here; the LAST one's final position is the
    # next-token distribution for the first generated token.
    grabbed = {}

    def _hook(_m, _inp, out):
        grabbed["logits"] = out.detach()[0, -1].float().cpu()

    h = model.lm_head.register_forward_hook(_hook)

    rows = []
    try:
        for depth in args.depths:
            prompt = build_prompt(tok, args.ctx, depth)
            sid = f"lf-{args.arm}-{depth}"
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
            w.active_session = sid
            grabbed.clear()
            w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0,
                       query_text="What is the secret passcode? Repeat it exactly.")
            lg = grabbed.get("logits")
            if lg is None:
                print(f"{args.arm} d={depth}: no logits captured", flush=True)
                continue
            rows.append({"arm": args.arm, "depth": depth, "logits": lg.tolist()})
            print(f"{args.arm} d={depth}: captured {lg.numel()} logits, "
                  f"top1={int(lg.argmax())}", flush=True)
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
    finally:
        h.remove()

    with open(args.json, "w") as f:
        json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--arm", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.arm:
        return run_one_arm(args)

    import subprocess
    import tempfile
    import torch

    tmp = tempfile.mkdtemp(prefix="dkv-logit-")
    per_arm = {}
    for arm in args.arms:
        out = os.path.join(tmp, f"{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--model", args.model, "--ctx", str(args.ctx), "--json", out,
               "--depths"] + [str(d) for d in args.depths]
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(out):
            print(f"{arm}: FAILED (exit {p.returncode})", flush=True)
            continue
        with open(out) as f:
            per_arm[arm] = {r["depth"]: torch.tensor(r["logits"]) for r in json.load(f)}

    if "dense" not in per_arm:
        raise SystemExit("no dense control -- nothing to compare against")

    print(f"\n== first-step logit fidelity vs DENSE "
          f"(n={len(args.depths)} prompts, ctx={args.ctx}) ==")
    print(f"{'arm':>12} {'top1 agree':>11} {'KL(dense||arm)':>15} "
          f"{'dense-top1 rank':>16} {'top5 overlap':>13}")
    for arm in args.arms:
        if arm not in per_arm:
            continue
        agree, kls, ranks, ov = 0, [], [], []
        n = 0
        for d, dl in per_arm["dense"].items():
            al = per_arm[arm].get(d)
            if al is None:
                continue
            n += 1
            pd = torch.softmax(dl, dim=-1)
            pa = torch.softmax(al, dim=-1)
            kls.append(float((pd * ((pd + 1e-12).log() - (pa + 1e-12).log())).sum()))
            dt = int(dl.argmax())
            agree += int(dt == int(al.argmax()))
            ranks.append(int((al > al[dt]).sum()))
            ov.append(len(set(torch.topk(dl, 5).indices.tolist())
                          & set(torch.topk(al, 5).indices.tolist())))
        if not n:
            continue
        print(f"{arm:>12} {agree:>7}/{n:<3} {sum(kls)/n:>15.5f} "
              f"{sum(ranks)/n:>16.2f} {sum(ov)/n:>12.1f}/5")
    print("\nRead every arm against BASELINE, not against dense: the")
    print("baseline gap is what compression already costs and is not the")
    print("feature's doing. An arm level with baseline has added nothing.")


if __name__ == "__main__":
    main()
