#!/usr/bin/env python3
"""Needle recall on a FINE depth grid, DKV against a dense control.

WHY THIS EXISTS
---------------
`validate_cuda_dkv.py` samples three depths (0.0, 0.5, 0.9) and reports 9/9.
`colab/needle_suite_cuda.py` samples 0.1/0.5/0.9 and reports a failure at
`8k@0.1`. Three points cannot answer "does CUDA capture the needle at ANY
depth", and the two suites disagree at the one depth only one of them tests.

Two things could each explain that failure and they need separating:

  TOKENISATION   niah_recall's needle is OMEGA-7741-DELTA, which Qwen splits
                 ' O'|'ME'|'GA'. `validate_cuda_dkv._assert_needle_unambiguous`
                 exists precisely because a partial-word needle makes recall a
                 coin flip on a small model for reasons that have nothing to do
                 with the KV cache (measured greedy top-2 margin 0.1875 logits).
                 The observed failure returns '7741-DELTA' -- missing exactly
                 the fragmenting prefix -- which is what that trap looks like.
  A REAL DEFECT  DKV genuinely loses the block at some depths.

So this sweeps a fine grid with a needle that passes the unambiguity check, and
runs a DENSE control at every point. Dense uses the same prompt and the same
answer test, so a depth where BOTH fail is the prompt or the model, and a depth
where only DKV fails is DKV.

READ THE DENSE ROW FIRST. A DKV failure at a depth dense also fails is not a
DKV failure.

USAGE
    python colab/needle_depth_sweep.py                       # 8k, 11 depths
    python colab/needle_depth_sweep.py --ctx 32768 --arms dkv dense
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCH = os.path.join(REPO, "benchmarks")

# The validator's needle: every token is a whole word, a digit or a separator.
# NOT niah_recall's OMEGA-7741-DELTA -- see the module docstring.
NEEDLE = "Falcon-9427-6183"


def _unambiguous(tok, needle):
    parts = [tok.decode([i]) for i in
             tok(" " + needle, add_special_tokens=False).input_ids]
    words = {w.lower() for w in needle.replace("-", " ").split()}
    bad = [p for p in parts if p.strip() and not p.strip().isdigit()
           and p.strip() != "-" and p.strip().lower() not in words]
    return bad, parts


def run_arm(args):
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    import torch
    import niah_recall

    # Same prompt builder as every other harness here, with the fragmenting
    # needle swapped for one that cannot be missed for tokenisation reasons.
    niah_recall.NEEDLE = NEEDLE
    niah_recall.NEEDLE_SENT = f"The secret passcode is {NEEDLE}."
    from niah_recall import QUESTION, build_prompt

    rows = []
    if args.arm == "dense":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="cuda").eval()
        bad, parts = _unambiguous(tok, NEEDLE)
        print(f"[needle] {NEEDLE!r} -> {parts} {'BAD ' + str(bad) if bad else 'OK'}",
              flush=True)
        for d in args.depths:
            prompt = build_prompt(tok, args.ctx, d)
            ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=24, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            comp = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ok = NEEDLE in comp
            rows.append({"depth": d, "ok": bool(ok), "blocks": 0, "text": comp[:60]})
            print(f"  [{'PASS' if ok else 'FAIL'}] dense d={d:.2f} -> {comp[:50]!r}",
                  flush=True)
    else:
        from serving.hf_dkv_wrapper import DKVHFWrapper
        w = DKVHFWrapper(model_id=args.model,
                         config={"quantization": None, "rank": 32,
                                 "block_size": 256, "micro_block_size": 256,
                                 "preset": "mid"})
        w.ensure_loaded()
        tok = w.tokenizer
        bad, parts = _unambiguous(tok, NEEDLE)
        print(f"[needle] {NEEDLE!r} -> {parts} {'BAD ' + str(bad) if bad else 'OK'}",
              flush=True)
        for d in args.depths:
            prompt = build_prompt(tok, args.ctx, d)
            sid = f"depth-{args.ctx}-{d}"
            w.active_session = sid
            out = w.generate(prompt, max_new_tokens=24, temperature=0.0,
                             top_p=1.0, repetition_penalty=1.0,
                             query_text=QUESTION)
            comp = out[len(prompt):] if out.startswith(prompt) else out[-200:]
            ok = NEEDLE in comp
            nb = 0
            try:
                sess = w.manager.sessions.get(sid) or {}
                nb = int(sum(sess.get("num_blocks") or []))
            except Exception:                                    # noqa: BLE001
                nb = -1
            rows.append({"depth": d, "ok": bool(ok), "blocks": nb,
                         "text": comp[:60]})
            print(f"  [{'PASS' if ok else 'FAIL'}] dkv   d={d:.2f} "
                  f"blocks={nb} -> {comp[:50]!r}", flush=True)
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass

    with open(args.json, "w") as f:
        json.dump(rows, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--arms", nargs="+", default=["dense", "dkv"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.arm:
        return run_arm(args)

    import tempfile
    tmp = tempfile.mkdtemp(prefix="dkv-depth-")
    res = {}
    for arm in args.arms:
        out = os.path.join(tmp, f"{arm}.json")
        print(f"\n== arm {arm} @ {args.ctx} ==", flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--model", args.model, "--ctx", str(args.ctx), "--json", out,
               "--depths"] + [str(d) for d in args.depths]
        p = subprocess.run(cmd, cwd=REPO)
        if p.returncode != 0 or not os.path.exists(out):
            print(f"{arm}: FAILED (exit {p.returncode})", flush=True)
            continue
        with open(out) as f:
            res[arm] = json.load(f)

    print(f"\n{'depth':>7} " + " ".join(f"{a:>7}" for a in args.arms))
    for i, d in enumerate(args.depths):
        cells = []
        for a in args.arms:
            r = res.get(a, [])
            cells.append("PASS" if i < len(r) and r[i]["ok"] else
                         ("FAIL" if i < len(r) else "-"))
        print(f"{d:>7.2f} " + " ".join(f"{c:>7}" for c in cells))
    for a in args.arms:
        r = res.get(a, [])
        print(f"{a}: {sum(1 for x in r if x['ok'])}/{len(r)}")
    print("\nA DKV failure at a depth DENSE also fails is not a DKV failure.")


if __name__ == "__main__":
    main()
