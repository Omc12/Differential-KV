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
import io
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



_NAT_FILES = ("ACTIVE_RUNTIME/nat_paper.txt", "benchmarks/berry_paper.txt",
              "benchmarks/random_features_paper.txt")


def _filler_text(kind):
    """The haystack. `repeat` is what every needle harness in this repo uses.

    THAT IS A PROBLEM AND IT INFLATES EVERY NEEDLE NUMBER HERE. niah_recall's
    FILLER is ONE sentence -- 38 unique words -- tiled to length. A random
    alphanumeric code dropped into that is a colossal outlier, so DKV's residual
    budget, which spends its slots on the WORST-RECONSTRUCTED tokens in each
    block, is all but guaranteed to spend one on the needle. Recall then measures
    "is the needle distinctive", which it is by construction, rather than "does
    the compressed representation retain it".

    `natural` fills with real papers from this repo (1544 unique words in the
    first alone), where the needle competes with genuinely distinctive tokens for
    a budget of 40 slots per block. That is the condition an outside benchmark
    puts DKV in, and it is the one where DKV is reported to degrade at early and
    mid depth while staying perfect late.
    """
    if kind == "repeat":
        return None
    parts = []
    for rel in _NAT_FILES:
        fp = os.path.join(REPO, rel)
        if os.path.exists(fp):
            parts.append(io.open(fp, encoding="utf-8", errors="ignore").read())
    if not parts:
        raise SystemExit("no natural filler files found")
    return "\n\n".join(parts)


def _build_natural(tok, ctx, depth, text, needle_sent, question):
    """Same shape as niah_recall.build_prompt, different haystack."""
    body = tok.encode(text, add_special_tokens=False)
    needle = tok.encode(needle_sent + "\n", add_special_tokens=False)
    q = tok.encode(question, add_special_tokens=False)
    budget = max(100, ctx - len(needle) - len(q) - 80)
    reps = budget // max(1, len(body)) + 1
    allf = (body * reps)[:budget]
    at = int(len(allf) * depth)
    p1, p2 = tok.decode(allf[:at]), tok.decode(allf[at:])
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + p1 + "\n" + needle_sent + "\n" + p2 +
            "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n")


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
    _nat = _filler_text(args.filler)
    if _nat is not None:
        _bp = build_prompt
        build_prompt = lambda t, c, d: _build_natural(          # noqa: E731
            t, c, d, _nat, niah_recall.NEEDLE_SENT, QUESTION)
    print(f"[filler] {args.filler}", flush=True)

    rows = []
    if args.arm == "dense":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="cuda").eval()
        bad, parts = _unambiguous(tok, NEEDLE)
        print(f"[needle] {NEEDLE!r} -> {parts} {'BAD ' + str(bad) if bad else 'OK'}",
              flush=True)
        from transformers import DynamicCache
        for d in args.depths:
            prompt = build_prompt(tok, args.ctx, d)
            ids = tok(prompt).input_ids
            # CHUNKED prefill plus a hand-written greedy loop. model.generate()
            # on a 32k prompt tries to allocate 46 GiB on a 12 GB card -- it
            # materialises the whole attention at once -- so the dense CONTROL
            # was the arm that could not run, at exactly the context where the
            # comparison matters most. Chunking is numerically free: attention is
            # causal, so the last token attends the same keys either way.
            try:
                cache = DynamicCache(config=model.config)
            except TypeError:
                cache = DynamicCache()
            step, gen = 512, []
            with torch.inference_mode():
                for i in range(0, len(ids), step):
                    ch = ids[i:i + step]
                    out = model(input_ids=torch.tensor([ch], device="cuda"),
                                position_ids=torch.tensor(
                                    [list(range(i, i + len(ch)))], device="cuda"),
                                past_key_values=cache, use_cache=True)
                cur, pos = int(out.logits[0, -1].argmax()), len(ids)
                for _ in range(24):
                    gen.append(cur)
                    if cur == tok.eos_token_id:
                        break
                    out = model(input_ids=torch.tensor([[cur]], device="cuda"),
                                position_ids=torch.tensor([[pos]], device="cuda"),
                                past_key_values=cache, use_cache=True)
                    pos += 1
                    cur = int(out.logits[0, -1].argmax())
            comp = tok.decode(gen, skip_special_tokens=True)
            del cache, out
            torch.cuda.empty_cache()
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
            # ISOLATE THE COMPLETION BY THE QUESTION MARKER, not by length.
            #
            # The needle is IN THE PROMPT, so every length-based slice is a trap
            # and this harness hit two of them: `out[-200:]` made two runs with
            # identical output disagree on PASS/FAIL depending on where the
            # window landed, and a token slice with a backward margin can catch
            # the prompt's OWN copy of the needle -- which at depth 1.0 sits
            # immediately before the question.
            #
            # The prompt ends with QUESTION, so everything after its LAST
            # occurrence is the completion and nothing else, at every depth.
            _cut = out.rfind(QUESTION)
            if _cut >= 0:
                comp = out[_cut + len(QUESTION):]
            elif out.startswith(prompt):
                comp = out[len(prompt):]
            else:
                comp = ""                    # cannot isolate -> do not guess
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
    ap.add_argument("--filler", default="repeat", choices=["repeat", "natural"],
                    help="repeat = one sentence tiled (what every harness here "
                         "does, and it makes the needle a guaranteed outlier); "
                         "natural = real papers, where the needle competes")
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
               "--filler", args.filler, "--depths"] + [str(d) for d in args.depths]
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

    # ── RATE PER DEPTH BAND ──────────────────────────────────────────────
    # One sample per depth cannot tell 100% from 50%, and DKV's reported
    # weakness is a RATE (50 / 21 / 100 across early / mid / late), not a
    # hard failure. Banding the fine grid is what makes those comparable.
    #
    # The mechanism to test: the recency window is EXACT and everything
    # before it is low-rank reconstructed with a small residual budget. If
    # that is the cause, `late` is perfect and `early`/`mid` are not, and
    # the gap WIDENS with context because the window is a shrinking
    # fraction of it (measured: 1048 exact tokens at 8k, ~3% of 32k).
    bands = (("early", 0.0, 0.34), ("mid", 0.34, 0.67), ("late", 0.67, 1.01))
    print("")
    print(f"{'arm':>7} " + " ".join(f"{n:>14}" for n, _, _ in bands))
    for a in args.arms:
        r = res.get(a, [])
        cells = []
        for _n, lo, hi in bands:
            sel = [x for x in r if lo <= x['depth'] < hi]
            cells.append(
                f"{100 * sum(1 for x in sel if x['ok']) / len(sel):.0f}% "
                f"({sum(1 for x in sel if x['ok'])}/{len(sel)})" if sel else '-')
        print(f"{a:>7} " + " ".join(f"{c:>14}" for c in cells))
    print("")
    print("The recency window is EXACT; everything before it is reconstructed.")
    print("If that is the mechanism, `late` is perfect and `early`/`mid` are not.")


if __name__ == "__main__":
    main()
