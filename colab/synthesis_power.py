"""Synthesis scoring with enough statistical power to resolve a difference.

WHY THIS EXISTS
---------------
`multifact_eval_cuda.py --tests synthesis` reports ONE number per config, from one
20-item draw (15 facts + 5 linked pairs) of one fixed document window. That number
moves 30 points when nothing changes but `DKV_RSVD_SEED`:

    ultra, rank 224, oversample 5:   seed 0 -> 63.3   seed 1 -> 33.3   seed 2 -> 50.0

so anything smaller than ~15 points is invisible to it. Two conclusions in this
project were built on single-seed readings and had to be retracted. This harness
exists so that does not happen a third time.

THREE THINGS IT FIXES
---------------------
1. ONE SAMPLE -> R REPLICATES. Each replicate re-prefills and re-scores, so the
   spread is measured rather than assumed.

2. DENSE HAD n=1 BY CONSTRUCTION. The document window and question were fixed, and
   dense has no randomised SVD, so dense could only ever produce one number -- there
   was nothing to compare a DKV distribution against. Replicates therefore vary the
   DOCUMENT WINDOW as well as the SVD seed, which gives both arms a real
   distribution over the same population of inputs.

3. UNPAIRED -> PAIRED. Every arm sees the SAME replicate list in the same order, so
   replicate r is the same document window for DKV and for dense. The statistic is
   the per-replicate DIFFERENCE, whose variance is far smaller than either arm's own
   spread -- the same reason bench_decode_paired.py can resolve effects that
   process-to-process comparison cannot.

Run one arm per process (model load dominates otherwise, and arms must not share a
wrapper -- multifact's own docstring records that sharing a session makes the same
config score differently). Then combine with --compare.

    python colab/synthesis_power.py --arm dkv   --reps 8 --out dkv.json
    python colab/synthesis_power.py --arm dense --reps 8 --out dense.json
    python colab/synthesis_power.py --compare dkv.json dense.json

Env: DKV_PRESET and any DKV_* knob apply to the dkv arm as usual.
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

# The probe words are rigorous_eval's and are drawn from the Random Features
# paper. Scoring them against a different document measures nothing.
FACTS = ["rahimi", "recht", "bochner", "fourier", "sinusoid", "hoeffding",
         "kernel", "random features", "svm", "regression", "gaussian",
         "least squares", "cvm", "forest", "randomly shifted"]
LINKED_PAIRS = [("rahimi", "recht"), ("bochner", "fourier"),
                ("hoeffding", "convergence"), ("randomly shifted", "binning"),
                ("random features", "kernel")]

QUESTION = ("\n\nSummarise the key technical contributions of this paper, naming "
            "the authors, the mathematical results it relies on, and the methods "
            "it is compared against.")


def score(out: str):
    o = out.lower()
    n_f = sum(1 for f in FACTS if f in o)
    n_l = sum(1 for a, b in LINKED_PAIRS if a in o and b in o)
    return (n_f / len(FACTS)) * 50.0 + (n_l / len(LINKED_PAIRS)) * 50.0, n_f, n_l


def replicates(reps: int):
    """The shared replicate list. Deterministic, so every arm gets the same one.

    Each replicate is (document window offset, SVD seed). The offset is what gives
    the dense arm a distribution; the SVD seed is what exposes DKV's compression
    variance. Varying both together measures the thing we actually care about --
    how a config behaves over inputs and draws, not at one lucky point.
    """
    return [(r * 512, r) for r in range(reps)]


def build_body(tok, ctx: int, offset: int):
    paper = os.path.join(_ROOT, "benchmarks", "random_features_paper.txt")
    if not os.path.exists(paper):
        raise SystemExit(f"corpus missing: {paper}")
    text = open(paper, encoding="utf-8", errors="ignore").read()
    ids = tok(text, add_special_tokens=False).input_ids
    win = ids[offset:offset + ctx]
    if len(win) < ctx:                       # wrap so every replicate is full length
        win = (ids + ids)[offset:offset + ctx]
    return tok.decode(win) + QUESTION


def chat(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def run_dkv(a, reps):
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
    cfg = {"mode": "fp16"}
    if os.environ.get("DKV_PRESET"):
        cfg["preset"] = os.environ["DKV_PRESET"]
    w = PyTorchDKVHFWrapper(model_id=a.model, config=cfg, device="cuda")
    w.ensure_loaded()
    tok = w.tokenizer
    out = []
    for i, (off, seed) in enumerate(reps):
        # _rsvd_omega reads DKV_RSVD_SEED at CALL time, so setting it here really
        # does change the projection for the prefill that follows.
        os.environ["DKV_RSVD_SEED"] = str(seed)
        for sid in list(getattr(w.manager, "decode_workspace", {}) or {}):
            try:
                w.manager.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
        body = build_body(tok, a.ctx, off)
        txt = w.generate(prompt=chat(tok, body), max_new_tokens=a.new,
                         temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        txt = txt.rsplit("assistant", 1)[-1]
        s, nf, nl = score(txt)
        out.append(s)
        print(f"  rep {i}: off={off} seed={seed} score={s:.1f} "
              f"(facts {nf}/15, links {nl}/5)", flush=True)
    return out


def run_dense(a, reps):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="cuda",
                                                 dtype=torch.float16).eval()
    out = []
    for i, (off, seed) in enumerate(reps):
        body = build_body(tok, a.ctx, off)
        ids = tok(chat(tok, body), return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            past = None
            for s0 in range(0, ids.shape[1], a.chunk):
                o = model(input_ids=ids[:, s0:s0 + a.chunk], past_key_values=past,
                          use_cache=True)
                past = o.past_key_values
            nxt = int(o.logits[:, -1, :].argmax())
            got = [nxt]
            for _ in range(a.new - 1):
                o = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                          past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[:, -1, :].argmax())
                if nxt == tok.eos_token_id:
                    break
                got.append(nxt)
        s, nf, nl = score(tok.decode(got, skip_special_tokens=True))
        out.append(s)
        print(f"  rep {i}: off={off} score={s:.1f} (facts {nf}/15, links {nl}/5)",
              flush=True)
        del past
        torch.cuda.empty_cache()
    return out


def summarise(name, xs):
    n = len(xs)
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("inf")
    print(f"{name}: mean={m:.1f}  sd={sd:.1f}  n={n}  "
          f"95%CI=[{m - half:.1f}, {m + half:.1f}]")
    return m, sd, half


def compare(pa, pb):
    a = json.load(open(pa))
    b = json.load(open(pb))
    xa, xb = a["scores"], b["scores"]
    if len(xa) != len(xb):
        raise SystemExit("arms have different replicate counts -- not pairable")
    print(f"\n=== {a['arm']} vs {b['arm']} ===")
    summarise(a["arm"], xa)
    summarise(b["arm"], xb)
    d = [p - q for p, q in zip(xa, xb)]
    n = len(d)
    md = statistics.mean(d)
    sdd = statistics.stdev(d) if n > 1 else 0.0
    # t critical at 95%, small-sample values; 2.0 is close enough above ~10
    tcrit = {2: 12.7, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
             9: 2.31, 10: 2.26}.get(n, 2.0)
    half = tcrit * sdd / math.sqrt(n) if n > 1 else float("inf")
    print(f"PAIRED mean_diff={md:+.2f}  95%CI=[{md - half:+.2f}, {md + half:+.2f}]"
          f"  resolution=+-{half:.2f} points")
    if (md - half) * (md + half) <= 0:
        print(f"VERDICT NO DIFFERENCE RESOLVABLE at n={n}. "
              f"To resolve a 5-point effect you need about "
              f"{max(2, math.ceil((1.96 * sdd / 5.0) ** 2))} replicates.")
    else:
        print(f"VERDICT {a['arm']} is {'ahead' if md > 0 else 'behind'} "
              f"by {abs(md):.1f} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["dkv", "dense"])
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--new", type=int, default=320)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.compare:
        compare(*a.compare)
        return
    if not a.arm:
        raise SystemExit("need --arm or --compare")

    reps = replicates(a.reps)
    label = a.arm if a.arm == "dense" else f"dkv/{os.environ.get('DKV_PRESET', 'mid')}"
    print(f"ARM {label}  model={a.model} ctx={a.ctx} reps={a.reps}", flush=True)
    xs = (run_dense if a.arm == "dense" else run_dkv)(a, reps)
    print()
    summarise(label, xs)
    if a.out:
        json.dump({"arm": label, "scores": xs, "reps": a.reps, "ctx": a.ctx},
                  open(a.out, "w"))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
