"""Run MLX's OWN NIAH bench through the CUDA DKV engine.

The 1.5B scores badly on validate_cuda_dkv's 9-case sweep while MLX is reported
at 9/9, but the two are not the same test: different needle
(OMEGA-7741-DELTA vs ZEBRA-4471-QUARTZ), different filler (MLX repeats ONE
4-sentence paragraph; the validator samples randomly from an 8-sentence pool),
and different scoring (substring vs normalised-alnum inside 24 tokens).

Repetitive filler is much easier for a compressed KV cache: near-duplicate blocks
reconstruct almost perfectly, so the needle's error towers over everything else
and wins residual slots without competition. Randomised filler makes it compete.

This holds the ENGINE fixed (CUDA DKV) and swaps only the BENCH, so whatever
changes is attributable to the bench and not to MLX-vs-CUDA. Prompt construction
is copied verbatim from tests/test_mlx_niah.py.

BENCH=mlx|dkv chooses the needle/filler/scoring; MODEL selects the model.
"""
import os
import sys
import random

ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))

os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
os.environ.setdefault("DKV_SYNC_COMPRESS", "1")   # determinism for a fidelity run
from serving.decode_config import BEST_DECODE_DEFAULTS  # noqa: E402
for _k, _v in BEST_DECODE_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
BENCH = os.environ.get("BENCH", "mlx")

# ── MLX bench (verbatim from tests/test_mlx_niah.py) ──────────────────────────
MLX_NEEDLE = "The secret passcode is OMEGA-7741-DELTA."
MLX_CODE = "OMEGA-7741-DELTA"
MLX_QUESTION = "What is the secret passcode? Repeat it exactly."
MLX_FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

# ── validator bench ──────────────────────────────────────────────────────────
DKV_CODE = "ZEBRA-4471-QUARTZ"
DKV_POOL = [
    "The morning fog rolled over the hills before the sun broke through the clouds.",
    "Researchers published a new dataset covering climate trends across five continents.",
    "The old library smelled of dust and aging paper, a comfort to regular visitors.",
    "Markets fluctuated throughout the week as investors weighed new economic data.",
    "A gentle breeze carried the scent of pine through the quiet mountain trail.",
    "The committee reviewed dozens of proposals before selecting a final design.",
    "Local farmers reported a strong harvest season despite the unpredictable weather.",
    "The orchestra rehearsed late into the evening, perfecting the final movement.",
]


def mlx_prompt(tok, target_tokens, depth):
    filler_toks = tok.encode(MLX_FILLER, add_special_tokens=False)
    needle_toks = tok.encode(MLX_NEEDLE + "\n", add_special_tokens=False)
    q_toks = tok.encode(MLX_QUESTION, add_special_tokens=False)
    budget = target_tokens - len(needle_toks) - len(q_toks) - 80
    if budget < 0:
        budget = 100
    repeats = (budget // len(filler_toks)) + 1
    all_filler = (filler_toks * repeats)[:budget]
    at = int(len(all_filler) * depth)
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            + tok.decode(all_filler[:at]) + "\n"
            + MLX_NEEDLE + "\n"
            + tok.decode(all_filler[at:]) + "\n\n"
            + MLX_QUESTION + "<|im_end|>\n<|im_start|>assistant\n")


def dkv_prompt(tok, n_filler, depth):
    filler = [random.choice(DKV_POOL) for _ in range(n_filler)]
    at = int(len(filler) * depth)
    needle = (f"Remember this important code: {DKV_CODE}. "
              "This is the only code you need to remember.")
    parts = filler[:at] + [needle] + filler[at:]
    parts.append("Question: What was the important code mentioned in this "
                 "text? Reply with only the code.")
    return tok.apply_chat_template([{"role": "user", "content": " ".join(parts)}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16"}, device="cuda")
    w.ensure_loaded()
    tok = w.tokenizer
    code = MLX_CODE if BENCH == "mlx" else DKV_CODE
    norm = lambda s: "".join(c for c in s.upper() if c.isalnum())  # noqa: E731

    random.seed(5)
    # Same nine (context, depth) points the validator sweeps, so only the
    # needle/filler/scoring differ.
    cases = [("2k", 2000, 200), ("8k", 8000, 800), ("32k", 32000, 2400)]
    n_pass = 0
    n_tot = 0
    for label, ctx_tokens, n_filler in cases:
        for depth in (0.0, 0.5, 0.9):
            if BENCH == "mlx":
                prompt = mlx_prompt(tok, ctx_tokens, depth)
            else:
                prompt = dkv_prompt(tok, n_filler, depth)
            ntok = len(tok(prompt).input_ids)
            outs = []
            for _ in range(3):
                r = w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                               top_p=1.0, repetition_penalty=1.0)
                outs.append(r.rsplit("assistant", 1)[-1].strip())
            hits = sum(norm(code) in norm(o) for o in outs)
            n_tot += 1
            n_pass += (hits == 3)
            print(f"BENCH={BENCH} {label}@{depth:.1f} ntok={ntok} "
                  f"recall={hits}/3 distinct={len(set(outs))} "
                  f"out={outs[0][:56]!r}", flush=True)
    print(f"TOTAL BENCH={BENCH} MODEL={MODEL}: {n_pass}/{n_tot}", flush=True)


if __name__ == "__main__":
    main()
