#!/usr/bin/env python3
"""Probe: is the fp16 fused-decode case flip a numerical defect or a coin-flip?

Background. `DKV_DECODE_FUSED_FP32` (read per route interval at
mlx_dkv_wrapper.py:5367) selects the storage dtype of the persistent fused
decode buffer. "1" (default) stores K/V fp32; "0" stores fp16, which is
~7 ms/token faster but was bisected on 2026-07-10 to return
'OMEGA-7741-Delta' instead of 'OMEGA-7741-DELTA' -- content right, case
flipped. The dial's comment records the flip as MULTI-needle only (single
needle passes 4/4 + depths 3/3), so this probe uses the multi-needle prompt.

The question this answers, and nothing more: at the first token where the two
arms disagree, is the fp32 arm's top-2 logit gap TINY or HEALTHY?

  tiny gap    -> the model is near-indifferent between the two casings. fp16 is
                 not corrupting a computation, it is losing a coin-flip that
                 fp32 happened to win. Not a numerical bug; take the speed.
  healthy gap -> fp16 moved the logits by more than the decision margin. Real
                 defect; go look at fp16 saturation near the -3e4 mask clamp
                 and the SDPA accumulation dtype.

Method. Both arms run in ONE process against ONE model load and ONE prompt,
greedy (temperature 0), repetition penalty off. Every decode step's logit
vector is captured by wrapping the module-level `_sanitize_logits`
(mlx_dkv_wrapper.py:133, called once per token from `sample_logits` at :7197)
-- no wrapper internals are patched, and the recorder delegates to the original
so sampling is unchanged. Sessions are cleared between arms because the fused
buffer is cached per route interval and would otherwise carry the first arm's
dtype into the second.

FINDING (2026-08-31, this Mac, Qwen2.5-1.5B-Instruct-4bit, 32k/block1024,
greedy, n=1 prompt -- single config, not a sweep):

  DKV_DECODE_FUSED_FP32 is DEAD CONFIG on the production default. Its only
  reader, _execute_decode_cache, is gated on DKV_DECODE_CACHE, which defaults
  to "0" (:2190) -- measured directly: 0 calls with the default, 1792 with the
  cache on. The fp16 arm's "~7 ms/token" is unavailable unless you first turn
  on a path that was deliberately reverted to off (:2185) for degrading recall.

  Measured cost of that precondition, one process, one prompt:
      cache=0 (default)  1/4 needles exact, clean <|im_end|> stop
      cache=1 fp32       0/4, degenerates into a repetition loop
      cache=1 fp16       0/4, degenerates into a repetition loop

  RETRACTED (2026-08-31), read this before citing the table above: those three
  cells all ran with DKV_RESIDUAL_QUANT=int4, which was then the default and is
  now known to destroy long-context recall on its own (0/48 vs fp16's 42/48 at
  ctx=20000, 12 randomised trials). Every arm here was crippled upstream of the
  dial, so this measures quantisation damage, not decode-cache staleness, and
  the "enabling the cache costs a needle" reading it originally carried is NOT
  supported. With the residual default corrected, cache=1 scores 4/4 at 16k.
  The decode-cache default needs its own clean re-measurement; nothing here
  speaks to it. What DOES survive is the coverage finding below: the dial is
  unreachable on the production default.

  The 2026-07-10 attribution of the case flip to fp16 is NOT supported here.
  In the SAME cache=0 run -- one dtype, no fused buffer at all -- the model
  emits 'OMEGA-7741-DELTA' in correct case and 'Kappa-4419-Gamma' flipped.
  A single buffer dtype cannot explain a casing difference between two needles
  inside one output. Casing tracks per-needle recall strength (KAPPA sits at
  depth 0.8, the weakest slot), not buffer numerics.

Usage:
    python3 benchmarks/probe_fp16_caseflip.py --block 1024 --ctx 32000

Note: --block 256 is a dead end on MLX (degenerates in ALL arms including the
default, so it isolates nothing). Use 1024.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

# Mirror the multi-needle harness's runtime configuration exactly.
os.environ["DKV_COMPRESSED_DECODE"] = "1"
os.environ.setdefault("DKV_MAX_RESIDUAL", "128")
# REQUIRED, and not the default: `_execute_decode_cache` -- the ONLY reader of
# DKV_DECODE_FUSED_FP32 -- is gated on `self._decode_cache` at
# mlx_dkv_wrapper.py:5453, and DKV_DECODE_CACHE defaults to "0" (:2190). Without
# this the dtype dial is inert and both arms are byte-identical for a reason that
# has nothing to do with fp16. The coverage gate below refuses to report a
# conclusion unless the branch is observed to run in both arms.
os.environ.setdefault("DKV_DECODE_CACHE", "1")

import serving.mlx_dkv_wrapper as W  # noqa: E402
from serving.mlx_dkv_wrapper import MLXDKVWrapper, MLXKVBlockManager  # noqa: E402

# The multi-needle probe set. Reproduced from benchmarks/run_multi_needle_mlx.py
# so this probe stands alone; NEEDLES[0] is the one that flipped case.
NEEDLES = [
    ("OMEGA-7741-DELTA", "The first secret passcode is OMEGA-7741-DELTA."),
    ("SIGMA-9923-BETA", "The second secret passcode is SIGMA-9923-BETA."),
    ("THETA-1105-ALPHA", "The third secret passcode is THETA-1105-ALPHA."),
    ("KAPPA-4419-GAMMA", "The fourth secret passcode is KAPPA-4419-GAMMA."),
]

QUESTION = "What are the four secret passcodes? List all of them clearly."

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)


def make_multi_needle_prompt(tokenizer, target_tokens, depths=(0.2, 0.4, 0.6, 0.8)):
    """Byte-for-byte the builder from run_multi_needle_mlx.py."""
    filler_toks = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_tok_list = [tokenizer.encode(s + "\n", add_special_tokens=False) for _, s in NEEDLES]
    question_toks = tokenizer.encode(QUESTION, add_special_tokens=False)
    overhead = sum(len(n) for n in needle_tok_list) + len(question_toks) + 80

    budget = max(100, target_tokens - overhead)
    repeats = (budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:budget]

    indices = sorted(int(len(all_filler) * d) for d in depths)
    parts, prev = [], 0
    for i, idx in enumerate(indices):
        parts.append(tokenizer.decode(all_filler[prev:idx]))
        parts.append("\n" + NEEDLES[i][1] + "\n")
        prev = idx
    parts.append(tokenizer.decode(all_filler[prev:]))

    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + "".join(parts) + "\n\n"
        + QUESTION + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class LogitRecorder:
    """Captures each decode step's logit vector, then delegates to the original.

    Installed over the module global `_sanitize_logits`. `sample_logits` in
    `generate()` resolves that name from module globals at CALL time, so this
    patch is picked up without touching the wrapper.
    """

    def __init__(self, original):
        self.original = original
        self.steps = []

    def __call__(self, logits, warn_owner=None):
        arr = np.asarray(logits, dtype=np.float64)
        self.steps.append(arr.copy())
        return self.original(logits, warn_owner)


class CoverageProbe:
    """Proof that the branch under test actually executed, and with which dtype.

    A null result ("both arms agree") is only meaningful if the fused decode
    path RAN and the two arms genuinely stored different dtypes. This wraps
    `MLXKVBlockManager._execute_decode_cache` to count calls and to read back
    the dtype of the fused K buffer the call left in `session["_cache_kv"]`.
    """

    def __init__(self):
        self.original = MLXKVBlockManager._execute_decode_cache
        self.calls = 0
        self.dtypes = set()

    def install(self):
        probe = self

        def wrapped(mgr_self, session, layer_idx, *a, **kw):
            probe.calls += 1
            out = probe.original(mgr_self, session, layer_idx, *a, **kw)
            ent = session.get("_cache_kv", {}).get(layer_idx)
            if ent is not None and "fk" in ent:
                probe.dtypes.add(str(ent["fk"].dtype))
            return out

        MLXKVBlockManager._execute_decode_cache = wrapped

    def uninstall(self):
        MLXKVBlockManager._execute_decode_cache = self.original


def run_arm(wrapper, prompt, fp32, max_new, tag, cache=True):
    """Run one arm end-to-end.

    `cache` selects DKV_DECODE_CACHE. It is latched onto the manager in
    __init__, so flipping the env alone is NOT enough -- the live attribute is
    reassigned here too, which is what makes a same-process three-arm
    comparison possible.

    Returns (completion_text, [logit vectors], [token ids], coverage).
    """
    os.environ["DKV_DECODE_FUSED_FP32"] = "1" if fp32 else "0"
    os.environ["DKV_DECODE_CACHE"] = "1" if cache else "0"
    wrapper.manager._decode_cache = bool(cache)

    sid = f"caseflip_{tag}"
    wrapper.manager.clear_session(sid)
    if hasattr(wrapper, "_session_token_ids"):
        wrapper._session_token_ids[sid] = []
    wrapper.active_session = sid
    # Loop-detector state is per-wrapper and sticky; clear it so arm B starts
    # from the same place arm A did.
    for attr in ("_mlx_loop_detected", "_mlx_loop_idx"):
        if hasattr(wrapper, attr):
            delattr(wrapper, attr)

    rec = LogitRecorder(W._sanitize_logits)
    cov = CoverageProbe()
    W._sanitize_logits = rec
    cov.install()
    try:
        wrapper.generate(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
    finally:
        W._sanitize_logits = rec.original
        cov.uninstall()

    # generate() returns prompt + completion, so its text CANNOT be used for
    # recall (every needle would "hit" off the echoed prompt). At temp=0 the
    # sampler is argmax, so the recorded per-step argmax IS the generated
    # sequence -- decode that instead.
    toks = [int(np.argmax(v)) for v in rec.steps]
    completion = wrapper.tokenizer.decode(toks)
    return completion, rec.steps, toks, cov


def top5(tokenizer, vec):
    """Return [(rank, token_id, repr(text), logit, softmax prob)] for the top 5."""
    idx = np.argsort(vec)[::-1][:5]
    shifted = vec - np.max(vec)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    out = []
    for r, i in enumerate(idx):
        try:
            txt = tokenizer.decode([int(i)])
        except Exception:
            txt = "<undecodable>"
        out.append((r, int(i), txt, float(vec[i]), float(probs[i])))
    return out


def print_table(tokenizer, vec, title):
    print(f"  {title}")
    print(f"    {'rank':<5} {'token_id':>9}  {'logit':>10}  {'prob':>8}  text")
    for r, tid, txt, lg, p in top5(tokenizer, vec):
        print(f"    {r:<5} {tid:>9}  {lg:>10.4f}  {p:>8.4f}  {txt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    print("=" * 78)
    print("fp16 fused-decode case-flip probe")
    print(f"  model={args.model}")
    print(f"  ctx={args.ctx}  block_size={args.block}  rank={args.rank}  max_new={args.max_new}")
    print("  greedy (temp=0), repetition_penalty=1.0, one model load, one prompt")
    print("=" * 78, flush=True)

    wrapper = MLXDKVWrapper(
        model_id=args.model,
        config={"rank": args.rank, "block_size": args.block},
    )
    prompt = make_multi_needle_prompt(wrapper.tokenizer, args.ctx)
    n_prompt = len(wrapper.tokenizer.encode(prompt))
    print(f"\nprompt tokens: {n_prompt}\n", flush=True)

    # Arm 0 is the production default. It exists to answer the question that
    # turns out to matter more than the dtype: what does the dial COST to reach?
    print("[arm 0] DKV_DECODE_CACHE=0  (production default — dial NOT reachable)", flush=True)
    text0, _, toks0, cov0 = run_arm(wrapper, prompt, True, args.max_new, "cache0", cache=False)
    hits0 = sum(1 for c, _ in NEEDLES if c in text0)
    print(f"  _execute_decode_cache calls={cov0.calls}  needles exact={hits0}/4")
    print(f"  -> {text0.strip()[:200]!r}\n", flush=True)

    print("[arm A] DKV_DECODE_FUSED_FP32=1  (fp32 buffer, requires DKV_DECODE_CACHE=1)", flush=True)
    text32, steps32, toks32, cov32 = run_arm(wrapper, prompt, True, args.max_new, "fp32")
    print(f"  -> {text32.strip()[:200]!r}\n", flush=True)

    print("[arm B] DKV_DECODE_FUSED_FP32=0  (fp16 buffer)", flush=True)
    text16, steps16, toks16, cov16 = run_arm(wrapper, prompt, False, args.max_new, "fp16")
    print(f"  -> {text16.strip()[:200]!r}\n", flush=True)

    # ---- COVERAGE GATE ----
    # Everything below is unreadable unless the branch under test actually ran
    # and the two arms really did store different dtypes.
    print("-" * 78)
    print("coverage (does the branch under test actually run?)")
    print(f"  arm A  _execute_decode_cache calls={cov32.calls:<6} fused K dtypes={sorted(cov32.dtypes) or 'NONE'}")
    print(f"  arm B  _execute_decode_cache calls={cov16.calls:<6} fused K dtypes={sorted(cov16.dtypes) or 'NONE'}")

    if cov32.calls == 0 or cov16.calls == 0:
        print("\nINVALID — the fused decode path never executed, so the dtype dial was")
        print("inert and this A/B tested nothing. Causes: DKV_DECODE_CACHE not 1, or")
        print("nb==0 (context too short to produce any compressed block).")
        sys.exit(2)

    if not cov32.dtypes or not cov16.dtypes:
        print("\nINVALID — _execute_decode_cache ran but never took the FUSED branch")
        print("(no 'fk' buffer was cached), so the dtype dial was never read.")
        print("Check DKV_DECODE_FUSED (must not be 0).")
        sys.exit(2)

    if cov32.dtypes == cov16.dtypes:
        print(f"\nINVALID — both arms stored the SAME dtype {sorted(cov32.dtypes)}. The")
        print("env flip did not reach the buffer, so any agreement below is an A/A")
        print("control, not an fp32-vs-fp16 result.")
        sys.exit(2)

    print(f"  arm 0  _execute_decode_cache calls={cov0.calls:<6} <- 0 means the dial is DEAD CONFIG by default")
    print("  gate PASSED — both dial arms ran the fused path and stored different dtypes.")

    # ---- cost of reaching the dial at all ----
    print("-" * 78)
    print("cost of enabling the decode cache (the dial's precondition)")
    h32 = sum(1 for c, _ in NEEDLES if c in text32)
    h16 = sum(1 for c, _ in NEEDLES if c in text16)
    print(f"  needles exact  cache=0 (default): {hits0}/4")
    print(f"                 cache=1 fp32     : {h32}/4")
    print(f"                 cache=1 fp16     : {h16}/4")
    if hits0 > max(h32, h16):
        print(f"  -> enabling the decode cache costs {hits0 - max(h32, h16)} needle(s). The dtype choice")
        print(f"     changes recall by {abs(h32 - h16)}. The precondition dominates the dial.")

    # ---- needle recall per arm (the outcome the dial was judged on) ----
    print("-" * 78)
    print("needle recall (exact, case-sensitive; measured on the COMPLETION only)")
    for code, _ in NEEDLES:
        a = code in text32
        b = code in text16
        ci = code.lower() in text16.lower()
        note = ""
        if not b and ci:
            note = "  <-- present but CASE DIFFERS"
        print(f"  {code:<18} fp32={'HIT ' if a else 'miss'}  fp16={'HIT ' if b else 'miss'}{note}")

    # ---- first divergence ----
    print("-" * 78)
    n = min(len(steps32), len(steps16))
    print(f"decode steps: fp32={len(steps32)}  fp16={len(steps16)}  (comparing first {n})")

    first = None
    for i in range(n):
        if toks32[i] != toks16[i]:
            first = i
            break

    if first is None:
        if len(steps32) == len(steps16):
            print("\nNO DIVERGENCE — both arms emitted identical tokens.")
            print("The flip does not reproduce at this ctx/block. The dial comment is")
            print("from 2026-07-10; try --block 256 --ctx 16000 before concluding.")
        else:
            print(f"\nNo divergence within the first {n} steps, but the arms ran to")
            print("different lengths — one stopped earlier. Re-run with a larger --max-new.")
        # Still report the largest logit perturbation seen, as a magnitude signal.
        d = max(float(np.max(np.abs(steps32[i] - steps16[i]))) for i in range(n))
        print(f"max |fp32-fp16| logit delta over {n} steps: {d:.6f}")
        return

    v32, v16 = steps32[first], steps16[first]
    order32 = np.argsort(v32)[::-1]
    gap32 = float(v32[order32[0]] - v32[order32[1]])
    # How far fp16 moved the logits of the two tokens actually in contention.
    a, b = int(order32[0]), int(order32[1])
    shift = float((v16[a] - v16[b]) - (v32[a] - v32[b]))

    print(f"\nFIRST DIVERGENCE at decode step {first}")
    print(f"  common prefix: {wrapper.tokenizer.decode(toks32[:first])!r}")
    print(f"  fp32 picked {toks32[first]} {wrapper.tokenizer.decode([toks32[first]])!r}")
    print(f"  fp16 picked {toks16[first]} {wrapper.tokenizer.decode([toks16[first]])!r}\n")

    print_table(wrapper.tokenizer, v32, "fp32 arm, top 5")
    print()
    print_table(wrapper.tokenizer, v16, "fp16 arm, top 5")

    print(f"\n  fp32 top-2 logit gap        : {gap32:.6f}")
    print(f"  fp16 shift of that same gap : {shift:+.6f}")
    print(f"  max |fp32-fp16| this step   : {float(np.max(np.abs(v32 - v16))):.6f}")

    print("\n" + "-" * 78)
    print("READING")
    if h32 == 0 and h16 == 0:
        print("  CAVEAT FIRST: both dial arms scored 0/4, so the model had already")
        print("  failed the task before this step. A divergence inside a degenerate")
        print("  tail says nothing about whether fp16 is safe for RECALL — read the")
        print("  cost table above instead, and treat the gap numbers below as")
        print("  describing junk tokens, not a needle decision.")
    if gap32 < 0.05:
        print(f"  fp32 top-2 gap is {gap32:.4f} < 0.05 — NEAR-INDIFFERENT.")
        print("  The model barely prefers its own answer here. fp16 is losing a")
        print("  coin-flip, not corrupting a computation. This is not a numerical")
        print("  defect, and the 2026-07-10 note reads as a misdiagnosis.")
    elif abs(shift) > gap32:
        print(f"  fp32 top-2 gap is {gap32:.4f} and fp16 moved it by {shift:+.4f} —")
        print("  the perturbation EXCEEDS the decision margin. Real numerical defect.")
        print("  Next: fp16 saturation near the -3e4 mask clamp (mlx_dkv_wrapper.py:5382)")
        print("  and the SDPA accumulation dtype.")
    else:
        print(f"  fp32 top-2 gap is {gap32:.4f} (healthy) but fp16 moved it only")
        print(f"  {shift:+.4f}, which does NOT cross the margin. The divergence at this")
        print("  step is therefore not explained by the logit shift alone — inspect")
        print("  whether an earlier step's state (not its argmax) already differed.")


if __name__ == "__main__":
    main()
