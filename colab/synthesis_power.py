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

try:                       # CUDA arms only; a Mac run never reaches them
    import torch
except ImportError:        # noqa: BLE001
    torch = None

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
    # Repeat enough times to cover offset+ctx. The corpus is ~8k tokens, so a
    # single doubling silently capped every ctx above ~16k at the corpus length --
    # a 32k run then measured the same 16k context and two configs came back
    # byte-identical, which is what exposed it. Compute the count instead.
    need = offset + ctx
    if len(ids) < need:
        ids = ids * (need // max(1, len(ids)) + 1)
    win = ids[offset:offset + ctx]
    assert len(win) == ctx, f"window {len(win)} != ctx {ctx}"
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
    # QUANT=nf4 for models that do not fit in fp16 -- a 7B is 14 GiB fp16 and the
    # card is 12. Both the config mode AND DKV_QUANTIZATION have to be set, the
    # same pair linkbench_cuda.py sets: the wrapper reads the config, the runtime
    # re-reads the env, and setting only one loads fp16 silently (the defect
    # fixed in 4385dd61, where quantization='int4' matched nothing).
    _q = os.environ.get("QUANT", "").lower()
    cfg = {"mode": "nf4" if _q == "nf4" else "fp16"}
    if _q == "nf4":
        os.environ["DKV_QUANTIZATION"] = "nf4"
    if os.environ.get("DKV_PRESET"):
        cfg["preset"] = os.environ["DKV_PRESET"]
    # BLOCK / RANK, matching the convention in linkbench_cuda.py and
    # multifact_eval_cuda.py. These are CONSTRUCTOR arguments, not environment
    # knobs the runtime re-reads -- without forwarding them here a sweep silently
    # runs the default every time and returns identical numbers for every arm,
    # which is exactly how this omission was caught.
    if os.environ.get("BLOCK"):
        cfg["micro_block_size"] = int(os.environ["BLOCK"])
    if os.environ.get("RANK"):
        cfg["rank"] = int(os.environ["RANK"])
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


# ── MLX arms ─────────────────────────────────────────────────────────────────
# The statistics (replicates, pairing, the CI, the power calculation) are shared
# with the CUDA arms above ON PURPOSE. The point of this harness is that both
# runtimes are judged by the same procedure; forking it into a second file is how
# two "comparable" numbers stop being comparable.
#
# The one genuine difference is the SEED VARIABLE. CUDA's randomised SVD reads
# DKV_RSVD_SEED, MLX's reads DKV_SVD_SEED (compress_mlx_block_batched), and both
# read it at CALL time so setting it per replicate really does redraw the
# projection for the prefill that follows. Vary it: temperature-0 replication is
# deterministic and therefore proves nothing -- the seed is the axis that has to
# move, and at a FIXED config it alone spans ~30 synthesis points.

def run_dkv_mlx(a, reps):
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
    cfg = {"preset": os.environ.get("DKV_PRESET", "mid")}
    # Constructor arguments, not environment knobs the runtime re-reads. Forgetting
    # to forward them silently runs the default for every arm of a sweep and
    # returns identical numbers, which is exactly how the CUDA omission was caught.
    if os.environ.get("BLOCK"):
        cfg["micro_block_size"] = int(os.environ["BLOCK"])
        cfg["block_size"] = int(os.environ["BLOCK"])
    if os.environ.get("RANK"):
        cfg["rank"] = int(os.environ["RANK"])
    w = MLXDKVWrapper(model_id=a.model, config=cfg)
    w.ensure_loaded()
    print(f"  [mlx dkv] preset={cfg['preset']} "
          f"rotated_pool={w.manager.rotated_pool} "
          f"block_size={w.manager.block_size}", flush=True)
    tok = w.tokenizer
    out = []
    for i, (off, seed) in enumerate(reps):
        os.environ["DKV_SVD_SEED"] = str(seed)
        for sid in list(getattr(w.manager, "sessions", {}) or {}):
            try:
                w.manager.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
        body = build_body(tok, a.ctx, off)
        txt = w.generate(prompt=chat(tok, body), max_new_tokens=a.new,
                         temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        txt = txt.rsplit("assistant", 1)[-1]
        sc, nf, nl = score(txt)
        out.append(sc)
        print(f"  rep {i}: off={off} seed={seed} score={sc:.1f} "
              f"(facts {nf}/15, links {nl}/5)", flush=True)
    return out


def run_dense_mlx(a, reps):
    """The control arm: mlx_lm with DKV not loaded.

    Replicates vary the DOCUMENT WINDOW here as well as the SVD seed. Dense has no
    randomised SVD, so a fixed window would give it n=1 BY CONSTRUCTION and leave
    the DKV distribution with nothing to be compared against — that was one of the
    three defects in the harness this file replaced.
    """
    from mlx_lm import generate as mlx_generate, load as mlx_load
    model, tok_w = mlx_load(a.model)
    # mlx_lm returns a TokenizerWrapper, which is NOT callable — build_body() calls
    # tok(text).input_ids. Unwrap for tokenisation, keep the wrapper for generate(),
    # so both arms are sized by the same builder on the same token counts.
    tok = getattr(tok_w, "_tokenizer", tok_w)
    out = []
    for i, (off, _seed) in enumerate(reps):
        body = build_body(tok, a.ctx, off)
        txt = mlx_generate(model, tok_w, prompt=chat(tok, body),
                           max_tokens=a.new, verbose=False)
        txt = txt.rsplit("assistant", 1)[-1]
        sc, nf, nl = score(txt)
        out.append(sc)
        print(f"  rep {i}: off={off} score={sc:.1f} (facts {nf}/15, links {nl}/5)",
              flush=True)
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
    ap.add_argument("--model", default="")
    ap.add_argument("--runtime", choices=["auto", "cuda", "mlx"], default="auto",
                    help="auto picks mlx on Apple silicon, cuda elsewhere.")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--new", type=int, default=320)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    runtime = a.runtime
    if runtime == "auto":
        runtime = "mlx" if sys.platform == "darwin" else "cuda"
    if not a.model:
        a.model = ("mlx-community/Qwen3.5-2B-4bit" if runtime == "mlx"
                   else "Qwen/Qwen3.5-2B")

    if a.compare:
        compare(*a.compare)
        return
    if not a.arm:
        raise SystemExit("need --arm or --compare")

    reps = replicates(a.reps)
    label = a.arm
    if a.arm != "dense":
        label = f"dkv/{os.environ.get('DKV_PRESET', 'mid')}"
        for _k in ("BLOCK", "RANK", "DKV_SVD_ENERGY", "DKV_TOPK_BLOCKS",
                   "DKV_PREFILL_CHUNK_SIZE"):
            if os.environ.get(_k):
                label += f" {_k}={os.environ[_k]}"
    # RECORD THE CONFIGURATION NEXT TO THE SCORE. A number compared against one
    # taken under a different runtime, preset or pool mode is not a comparison.
    if runtime == "mlx" and a.arm != "dense":
        label += f" rotated_pool={os.environ.get('DKV_ROTATED_POOL', '1')}"
    print(f"ARM {label}  runtime={runtime} model={a.model} ctx={a.ctx} "
          f"reps={a.reps}", flush=True)
    _runners = {("cuda", "dense"): run_dense, ("cuda", "dkv"): run_dkv,
                ("mlx", "dense"): run_dense_mlx, ("mlx", "dkv"): run_dkv_mlx}
    xs = _runners[(runtime, a.arm)](a, reps)
    print()
    summarise(label, xs)
    if a.out:
        json.dump({"arm": label, "scores": xs, "reps": a.reps, "ctx": a.ctx},
                  open(a.out, "w"))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
