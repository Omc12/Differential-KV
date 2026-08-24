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
    # `dense_arm` is DKV with compressed DECODE off. It is NOT a dense control:
    # the prompt is still read through DKV's block-sparse PREFILL, so it shares
    # every prefill-side behaviour with `baseline` and can only ever measure the
    # decode half. It was called `dense` here for months, and the 10.579 KL it
    # produced was read as a DKV defect without anything ruling out the reverse.
    # The MLX port hit the same trap and replaced its control with plain mlx_lm
    # (logit_fidelity_mlx.py:60) -- this is the CUDA equivalent.
    "dense_true": {},          # handled specially: plain HF, DKV never loaded
    "dense_arm":  {"DKV_COMPRESSED_DECODE": "0"},
    "baseline":   {},
    # Bisect arms for the compressed-decode gap. `allblocks` attends EVERY
    # compressed block (DKV_TOPK_BLOCKS=0, query_router.py:983) with one chunk,
    # so the reduction path is unchanged and only the ROUTED SET differs.
    # Reading, decided first: if it lands near dense_arm/dec the defect is
    # routing; if it stays near baseline/dec the defect is the reconstruction
    # math on compressed blocks and routing is exonerated.
    "allblocks":  {"DKV_TOPK_BLOCKS": "0", "DKV_BLOCKS_PER_CHUNK": "256"},
    # Capacity sweep. A correct low-rank reconstruction gets BETTER with more
    # rank and with more exact residual rows. Reading, decided first: if the
    # decode gap shrinks with capacity it is approximation error and the
    # operating point is the story; if it is flat, extra capacity is buying
    # nothing, which no correct reconstruction does -- that is a defect.
    "rank96":     {},
    "rank128":    {},
    "resid200":   {"DKV_MAX_RESIDUAL": "200"},
    # Two INDEPENDENT implementations of the same compressed read: `noremat`
    # takes the Triton project-then-attend kernel (s = s_anchor + (q.V_K).U.scale)
    # instead of materialise-then-SDPA. Reading, decided first: if both land at
    # the same KL the defect is in their SHARED INPUTS -- the pool contents or
    # the routed gather -- and neither attention implementation is at fault.
    # If they split, the defect is inside whichever one is worse.
    "noremat":    {"DKV_REMAT_CACHE": "0"},
    # remat, but with its attention run in fp32 instead of q.dtype. Reading,
    # decided first: if this lands near `noremat` the root is PRECISION -- fp16
    # cannot resolve a softmax whose scores reach ~1e4; if it stays near
    # `baseline` precision is exonerated and the difference is structural.
    # remat is CORRECT under the shipped serving defaults and wrong without
    # them. These arms bisect which of those defaults it silently depends on.
    "remat_dc":   {"DKV_DECODE_CACHE": "1"},
    "remat_bias": {"DKV_SPARSE_BIAS": "auto"},
    "remat_gsd":  {"DKV_GRAPH_SAFE_DECODE": "1"},
    # And the rotational frame. An unrotated pool rotates on the read path
    # instead of the write path, so a frame error shows up as a LARGE change
    # here while a genuine approximation error barely moves.
    "unrot":      {"DKV_ROTATED_POOL": "0", "DKV_DECODE_CACHE": "1"},
    "basis0.50":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.50"},
    "basis0.25":  {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.25"},
    "basis0.125": {"DKV_SHARED_BASIS": "1", "DKV_SHARED_BASIS_FRAC": "0.125"},
}

# Arms that change the WRAPPER CONFIG rather than the environment.
ARM_CONFIG = {
    "rank96":  {"rank": 96},
    "rank128": {"rank": 128},
}
_ARM_KEYS = ("DKV_SHARED_BASIS", "DKV_SHARED_BASIS_FRAC", "DKV_COMPRESSED_DECODE",
             "DKV_TOPK_BLOCKS", "DKV_BLOCKS_PER_CHUNK", "DKV_MAX_RESIDUAL",
             "DKV_REMAT_CACHE", "DKV_ROTATED_POOL", "DKV_DECODE_CACHE",
             "DKV_REMAT_FP32")


def _run_dense_true(args):
    """The real control: plain HF transformers, DKV never imported or loaded.

    The first-step next-token logits are the last row of a causal forward over
    the prompt -- exactly what the DKV arm's lm_head hook captures, since at
    step 1 with temperature 0 and repetition_penalty 1.0 every transform ahead
    of that hook is the identity.

    Fed through a DynamicCache in chunks: the lm_head runs on EVERY position,
    so a single-shot 8k forward over a 151,936-token vocab materialises
    8192*151936*2 B = 2.5 GB of logits on top of the weights. Chunking is
    numerically free -- attention is causal, so the last token attends the same
    keys either way -- and `position_ids` is passed explicitly so the chunked
    run is positionally identical to the single-shot one.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    sys.path.insert(0, BENCH)
    from niah_recall import build_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"[dense control] plain transformers, DKV NOT loaded ({args.model})",
          flush=True)

    rows = []
    for depth in args.depths:
        prompt = build_prompt(tok, args.ctx, depth)
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        try:
            cache = DynamicCache(config=model.config)
        except TypeError:
            cache = DynamicCache()
        step = 512
        with torch.inference_mode():
            for i in range(0, len(ids), step):
                ch = ids[i:i + step]
                out = model(input_ids=torch.tensor([ch], device="cuda"),
                            position_ids=torch.tensor(
                                [list(range(i, i + len(ch)))], device="cuda"),
                            past_key_values=cache, use_cache=True)
        lg = out.logits[0, -1].float().cpu()
        # ONE greedy decode step, so the `/dec` rows have a control at the SAME
        # position. Without it the decode capture was being scored against the
        # first-token control -- token N+2 vs token N+1, two different queries
        # over two different prefixes, which is what produced the 10.579 that
        # was read as a DKV defect.
        nxt = int(lg.argmax())
        with torch.inference_mode():
            out2 = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                         position_ids=torch.tensor([[len(ids)]], device="cuda"),
                         past_key_values=cache, use_cache=True)
        lg2 = out2.logits[0, -1].float().cpu()
        del cache, out, out2
        torch.cuda.empty_cache()
        rows.append({"arm": args.arm, "depth": depth, "blocks": 0,
                     "logits": lg.tolist(), "logits_decode": lg2.tolist(),
                     "tok1": nxt})
        print(f"dense_true d={depth}: {len(ids)} tok, captured {lg.numel()} "
              f"logits, top1={nxt}, step2_top1={int(lg2.argmax())}", flush=True)

    with open(args.json, "w") as f:
        json.dump(rows, f)


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
    if args.arm == "dense_true":
        return _run_dense_true(args)
    import torch
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt

    _cfg = {"quantization": None, "rank": 32, "block_size": 256,
            "micro_block_size": 256, "preset": "mid"}
    _cfg.update(ARM_CONFIG.get(args.arm, {}))
    print(f"[arm {args.arm}] config={_cfg} env="
          f"{ {k: os.environ[k] for k in _ARM_KEYS if k in os.environ} }", flush=True)
    w = DKVHFWrapper(model_id=args.model, config=_cfg)
    w.ensure_loaded()
    tok, model = w.tokenizer, w.model

    # Capture the lm_head output. generate() runs chunked prefill and THEN a
    # decode forward, so several calls land here.
    #
    # THE LAST CALL IS THE WRONG ONE. The decode loop (hf_dkv_wrapper.py:1836)
    # samples token 1 from the logits prefill already produced and then forwards
    # again, so even at max_new_tokens=1 an extra decode step runs. Taking the
    # last call therefore captured the distribution for token N+2 while the
    # dense control captured token N+1 -- two different queries over two
    # different prefixes. That is not a fidelity measurement at all, and it is
    # the same artifact probe_layer_output_diff.py had to fix on the layer side.
    #
    # The first-generated-token distribution is the last position of the LAST
    # PREFILL CALL: prefill chunks have L > 1, decode steps have L == 1.
    # The lm_head OUTPUT cannot tell the two apart: this stack slices hidden
    # states to the final position before the head, so a 1024-token prefill
    # chunk and a decode step both arrive as L == 1 (measured: 9 calls at 8k,
    # all shape[1]==1). Discriminate on the MODEL's input_ids instead, which is
    # the only place the distinction survives.
    grabbed = {}
    trace = []
    cur = {"L": 0}

    def _pre(_m, _a, kw):
        ii = kw.get("input_ids")
        if ii is None and _a:
            ii = _a[0]
        cur["L"] = int(ii.shape[1]) if ii is not None and hasattr(ii, "shape") else 0

    def _hook(_m, _inp, out):
        t = out.detach()
        L = cur["L"]
        trace.append(L)
        if L > 1:                       # prefill chunk -- keep the LAST one
            grabbed["prefill"] = t[0, -1].float().cpu()
        else:                           # decode step -- keep the FIRST one
            grabbed.setdefault("decode", t[0, -1].float().cpu())

    hp = model.register_forward_pre_hook(_pre, with_kwargs=True)
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
            trace.clear()
            w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0,
                       query_text="What is the secret passcode? Repeat it exactly.")
            lg = grabbed.get("prefill")
            _alt = grabbed.get("decode")
            print(f"    lm_head calls: {trace[:6]}{'...' if len(trace) > 6 else ''} "
                  f"(n={len(trace)}, prefill={sum(1 for x in trace if x > 1)}, "
                  f"decode={sum(1 for x in trace if x == 1)})", flush=True)
            if lg is None:
                print(f"{args.arm} d={depth}: no logits captured", flush=True)
                continue
            # ENGAGEMENT READOUT. Without it a small KL is ambiguous between
            # "DKV tracks dense" and "DKV never compressed anything" -- the CUDA
            # 4k arm once reported pool 0.0 MB for exactly that reason.
            try:
                sess = w.manager.sessions.get(sid) or {}
                nb = int(sum(sess.get("num_blocks") or []))
            except Exception:                                    # noqa: BLE001
                nb = -1
            rows.append({"arm": args.arm, "depth": depth, "blocks": nb,
                         "logits": lg.tolist(), "tok1": int(lg.argmax()),
                         "logits_decode": _alt.tolist() if _alt is not None else None})
            print(f"{args.arm} d={depth}: captured {lg.numel()} logits, "
                  f"top1={int(lg.argmax())}, compressed_blocks={nb}", flush=True)
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
    finally:
        h.remove()
        hp.remove()

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
    blocks = {}
    tok1 = {}
    report_arms = []
    for arm in args.arms:
        report_arms.append(arm)
        out = os.path.join(tmp, f"{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--model", args.model, "--ctx", str(args.ctx), "--json", out,
               "--depths"] + [str(d) for d in args.depths]
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(out):
            print(f"{arm}: FAILED (exit {p.returncode})", flush=True)
            continue
        with open(out) as f:
            _rows = json.load(f)
        per_arm[arm] = {r["depth"]: torch.tensor(r["logits"]) for r in _rows}
        blocks[arm] = [r.get("blocks", 0) for r in _rows]
        # The decode-step capture is kept as a SEPARATE pseudo-arm so the two
        # can never be silently swapped again. It answers a different question
        # (token N+2 vs token N+1) and is reported, not compared.
        if any(r.get("logits_decode") for r in _rows):
            per_arm[arm + "/dec"] = {r["depth"]: torch.tensor(r["logits_decode"])
                                     for r in _rows if r.get("logits_decode")}
            blocks[arm + "/dec"] = [r.get("blocks", 0) for r in _rows]
            report_arms.append(arm + "/dec")
        tok1[arm] = {r["depth"]: r.get("tok1") for r in _rows}

    ref = "dense_true" if "dense_true" in per_arm else "dense_arm"
    if ref not in per_arm:
        raise SystemExit("no dense control -- nothing to compare against")

    print(f"\n== first-step logit fidelity vs DENSE "
          f"(n={len(args.depths)} prompts, ctx={args.ctx}) ==")
    print(f"{'arm':>16} {'top1 agree':>11} {'KL(dense||arm)':>15} "
          f"{'dense-top1 rank':>16} {'top5 overlap':>13} {'blocks':>8}")
    for arm in report_arms:
        if arm not in per_arm:
            continue
        # A `/dec` row is only comparable if BOTH sides decoded from the same
        # step-1 token; otherwise the two prefixes differ and the row measures
        # divergence, not fidelity.
        # A `/dec` row must be scored against the `/dec` CONTROL -- the same
        # token position. Scoring it against the step-1 control is precisely the
        # defect this harness used to have, and dense_true/dec is kept as a
        # standing self-check on exactly that: it contains no compression of any
        # kind, so anything but ~0 there means the positions are misaligned
        # again and NO row in this table can be believed.
        my_ref = ref + "/dec" if arm.endswith("/dec") else ref
        if my_ref not in per_arm:
            continue
        if arm.endswith("/dec"):
            base = arm[:-4]
            bad = [d for d, t in tok1.get(base, {}).items()
                   if t is not None and tok1.get(ref, {}).get(d) != t]
            if bad:
                print(f"{arm:>16}   NOT COMPARABLE — step-1 token differs from "
                      f"{ref} at depths {bad}; the prefixes are not the same.")
                continue
        agree, kls, ranks, ov = 0, [], [], []
        n = 0
        for d, dl in per_arm[my_ref].items():
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
        nb = blocks.get(arm)
        nbs = "-" if not nb else f"{sum(nb)/len(nb):.0f}"
        print(f"{arm:>16} {agree:>7}/{n:<3} {sum(kls)/n:>15.5f} "
              f"{sum(ranks)/n:>16.2f} {sum(ov)/n:>12.1f}/5 {nbs:>8}")
    print("\nRead every arm against BASELINE, not against dense: the")
    print("baseline gap is what compression already costs and is not the")
    print("feature's doing. An arm level with baseline has added nothing.")
    print("\nThe */dec rows are the SECOND token, scored against the dense")
    print("control at that SAME position. dense_true/dec is the self-check:")
    print("it has no compression in it anywhere, so it must read ~0. If it")
    print("does not, the positions are misaligned and no row here means")
    print("anything -- which is exactly how this harness reported KL 10.579")
    print("for a defect that did not exist.")


if __name__ == "__main__":
    main()
