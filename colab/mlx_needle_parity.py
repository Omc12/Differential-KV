#!/usr/bin/env python3
"""Run validate_cuda_dkv.py's needle suite against the MLX DKV runtime.

WHY
---
The CUDA dense control settles that every remaining DKV failure is a DKV bug:
plain HF attention gets 3/3 on 2k@0.0, 8k@0.5 and 8k@0.9 where CUDA DKV gets
token salad, 1/3 and (before the bias fix) 0/3. So "the model can't do it" and
"the prompt is hard" are both dead.

What that control CANNOT say is which STAGE of DKV is at fault. MLX runs the
same algorithm and is the known-good reference, so running the IDENTICAL test on
it splits the remaining failures into two very different piles:

  MLX passes, CUDA fails  -> a CUDA implementation bug. Keep bisecting code.
  BOTH fail               -> the ALGORITHM loses this needle, on both runtimes.
                             No amount of CUDA-vs-MLX code reading will fix it,
                             and every hour spent diffing the two is wasted.

That distinction has never been measured. It has been assumed -- in both
directions, at different times -- and the parity harness has already caught
three assumptions of exactly this shape.

The context builder here is COPIED VERBATIM from validate_cuda_dkv.py (same
NEEDLE, same random.seed(5), same filler pool, same depth arithmetic, same
question) so the prompts are token-for-token identical. Any divergence in the
generator would make the comparison meaningless, which is the whole point of it.

    python colab/mlx_needle_parity.py            # 2k + 8k
    python colab/mlx_needle_parity.py --long     # + 32k
    python colab/mlx_needle_parity.py --dense    # MLX dense control
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

NEEDLE = "ZEBRA-4471-QUARTZ"

# ── verbatim from validate_cuda_dkv.py:189-218 ───────────────────────────────
FILLER = [
    "The morning fog rolled over the hills before the sun broke through the clouds.",
    "Researchers published a new dataset covering climate trends across five continents.",
    "The old library smelled of dust and aging paper, a comfort to regular visitors.",
    "Markets fluctuated throughout the week as investors weighed new economic data.",
    "A gentle breeze carried the scent of pine through the quiet mountain trail.",
    "The committee reviewed dozens of proposals before selecting a final design.",
    "Local farmers reported a strong harvest season despite the unpredictable weather.",
    "The orchestra rehearsed late into the evening, perfecting the final movement.",
]


def build(n_filler, depth=0.0):
    filler = [random.choice(FILLER) for _ in range(n_filler)]
    at = int(len(filler) * depth)
    needle = (f"Remember this important code: {NEEDLE}. "
              "This is the only code you need to remember.")
    parts = filler[:at] + [needle] + filler[at:]
    parts.append("Question: What was the important code mentioned in this "
                 "text? Reply with only the code.")
    return " ".join(parts)
# ─────────────────────────────────────────────────────────────────────────────


def norm(s):
    return "".join(c for c in s.upper() if c.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="add the ~32k cases")
    ap.add_argument("--quick", action="store_true", help="2k only")
    ap.add_argument("--dense", action="store_true", help="MLX dense control")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    print("=" * 62)
    print(f"  MLX DKV needle suite — {args.model}")
    print(f"  same NEEDLE / seed / filler / depths as validate_cuda_dkv.py")
    print("=" * 62)

    # Apply the SAME shipped defaults the CUDA validator applies. Without this
    # the two runs differ by configuration as well as by runtime, and the
    # comparison answers nothing. setdefault, so an explicit env still wins.
    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    if not args.dense:
        try:
            from serving.decode_config import BEST_DECODE_DEFAULTS
            for k, v in BEST_DECODE_DEFAULTS.items():
                os.environ.setdefault(k, v)
            print("  [config] " + ", ".join(
                f"{k}={os.environ[k]}" for k in BEST_DECODE_DEFAULTS))
        except Exception as e:                                    # noqa: BLE001
            print(f"  [config] could not apply serving defaults: {e}")

    if args.dense:
        from mlx_lm import generate as mlx_generate, load as mlx_load
        model, tok = mlx_load(args.model)
        print("  [dense control] mlx_lm, DKV NOT loaded")

        class W:
            tokenizer = tok

            def generate(self, prompt, max_new_tokens=24, **kw):
                return mlx_generate(model, tok, prompt=prompt,
                                    max_tokens=max_new_tokens, verbose=False)
        w = W()
    else:
        from serving.mlx_dkv_wrapper import MLXDKVWrapper
        w = MLXDKVWrapper(model_id=args.model, config={"preset": "mid"})

    cases = [("2k", 200, 0.0), ("2k", 200, 0.5), ("2k", 200, 0.9)]
    if not args.quick:
        cases += [("8k", 800, 0.0), ("8k", 800, 0.5), ("8k", 800, 0.9)]
    if args.long:
        cases += [("32k", 2400, 0.0), ("32k", 2400, 0.5), ("32k", 2400, 0.9)]

    random.seed(5)                    # seeded ONCE, as the CUDA validator does
    failed = []
    for label, n_filler, depth in cases:
        name = f"{label}@depth{depth:.1f}"
        ctx = build(n_filler, depth)
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "user", "content": ctx}], tokenize=False,
            add_generation_prompt=True)
        ntok = len(w.tokenizer.encode(prompt))

        outs = []
        for _ in range(args.repeat):
            r = w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                           top_p=1.0, repetition_penalty=1.0)
            outs.append(r.rsplit("assistant", 1)[-1].strip())

        hits = sum(norm(NEEDLE) in norm(o) for o in outs)
        distinct = len(set(outs))
        ok = hits == args.repeat
        det = distinct == 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({ntok} tok) recall "
              f"{hits}/{args.repeat}  distinct={distinct}  {outs[0][:58]!r}")
        if not ok:
            failed.append(name)
        if not det:
            failed.append(f"{name} determinism")

    print("\n" + "=" * 62)
    if failed:
        print(f"  FAILED ({len(failed)}): " + ", ".join(failed))
    else:
        print("  ALL PASS")
    print("=" * 62)
    print("\n  Compare case-by-case with the CUDA run. A case MLX passes and")
    print("  CUDA fails is a CUDA bug worth bisecting; a case BOTH fail is an")
    print("  algorithm limit, and diffing the two runtimes will never fix it.")


if __name__ == "__main__":
    main()
