#!/usr/bin/env python3
"""Per-layer attention-output diff for MLX: DKV vs its OWN dense baseline.

WHY THIS AND NOT THE CUDA-vs-DENSE DIFF
---------------------------------------
The CUDA probe compared DKV against dense and found layer-3 cosine ~0.28 -- in
BOTH the failing case AND the passing one (32k@0.5 measured 0.27598 and PASSES;
32k@0.9 measured 0.29126 and fails). The passing case was actually WORSE. So
"how far is DKV from dense" is not the question: DKV is a coarse approximation
everywhere, and recall does not depend on reproducing dense's output vector.

The question is "where does CUDA differ from the implementation that WORKS".
MLX passes all nine cases. So run the SAME measurement on MLX and read the gap:

    CUDA   cos(DKV_cuda, dense_cuda) @ layer3 = 0.291 (fail) / 0.276 (pass)
    MLX    cos(DKV_mlx,  dense_mlx ) @ layer3 = ?

Each runtime is compared against ITS OWN dense baseline, which is what makes
this valid despite MLX being the 4-bit build and CUDA fp16: the weights differ,
so raw vectors are not comparable across runtimes, but "how much does DKV
deviate from dense, in this runtime" is.

  MLX ~0.28 too  -> that deviation is inherent to the DKV algorithm and is NOT
                    the bug. Stop looking at attention outputs entirely.
  MLX ~0.9+      -> CUDA's decode is FAR coarser than MLX's on identical inputs,
                    and that gap is the bug. It would also explain why CUDA only
                    fails where the needle's margin over the filler is thinnest.

MLX IS NOT MODIFIED. Each layer's attn module is wrapped by a recording proxy
from this script; mlx_dkv_wrapper.py is untouched.

    python colab/probe_mlx_layer_output_diff.py --depth 0.9
    python colab/probe_mlx_layer_output_diff.py --depth 0.5   # passing control
"""
import argparse
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

NEEDLE = "ZEBRA-4471-QUARTZ"
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


def build(n_filler, depth):
    filler = [random.choice(FILLER) for _ in range(n_filler)]
    at = int(len(filler) * depth)
    needle = (f"Remember this important code: {NEEDLE}. "
              "This is the only code you need to remember.")
    parts = filler[:at] + [needle] + filler[at:]
    parts.append("Question: What was the important code mentioned in this "
                 "text? Reply with only the code.")
    return " ".join(parts)


def make_ctx(label, depth):
    """Replay the validator's FULL nine-case RNG sequence -- it seeds once, so a
    case built in isolation is a DIFFERENT PROMPT."""
    cases = [("2k", 200, 0.0), ("2k", 200, 0.5), ("2k", 200, 0.9),
             ("8k", 800, 0.0), ("8k", 800, 0.5), ("8k", 800, 0.9),
             ("32k", 2400, 0.0), ("32k", 2400, 0.5), ("32k", 2400, 0.9)]
    random.seed(5)
    ctx = None
    for lab, n, d in cases:
        c = build(n, d)
        if lab == label and abs(d - depth) < 1e-9:
            ctx = c
    assert ctx is not None
    return ctx


class _Rec:
    """Proxy around an attention module: forwards the call, records the output
    for the LAST position. Works whether DKV patches the class or the instance,
    because the layer still does `self.self_attn(...)` and reaches this object."""

    def __init__(self, inner, store, idx):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_idx", idx)

    def __call__(self, *a, **kw):
        import mlx.core as mx
        out = self._inner(*a, **kw)
        t = out[0] if isinstance(out, tuple) else out
        try:
            v = t.reshape(-1, t.shape[-1])[-1]
            self._store[self._idx] = mx.array(v).astype(mx.float32)
        except Exception:                                        # noqa: BLE001
            pass
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)


def _wrap(layers, store):
    n = 0
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or isinstance(attn, _Rec):
            continue
        layer.self_attn = _Rec(attn, store, i)
        n += 1
    return n


def _unwrap(layers):
    for layer in layers:
        a = getattr(layer, "self_attn", None)
        if isinstance(a, _Rec):
            layer.self_attn = object.__getattribute__(a, "_inner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--label", default="32k")
    ap.add_argument("--depth", type=float, default=0.9)
    args = ap.parse_args()

    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    from serving.decode_config import BEST_DECODE_DEFAULTS
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)

    import mlx.core as mx
    from mlx_lm import load as mlx_load

    ctx = make_ctx(args.label, args.depth)

    # ── Pass A: DENSE (plain mlx_lm, DKV never imported into this model) ──
    model, tok = mlx_load(args.model)
    prompt = tok.apply_chat_template([{"role": "user", "content": ctx}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt)
    print(f"[probe] {args.label}@depth{args.depth} — {len(ids)} tok")

    dense = {}
    layers = model.model.layers if hasattr(model, "model") else model.layers
    print(f"[probe] dense: wrapped {_wrap(layers, dense)} attention layers")
    out = model(mx.array([ids]))
    mx.eval(out)
    print(f"[probe] dense next token: {tok.decode([int(mx.argmax(out[0, -1]).item())])!r}")
    _unwrap(layers)
    del model
    import gc; gc.collect()

    # ── Pass B: MLX DKV, same prompt ──
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
    w = MLXDKVWrapper(model_id=args.model, config={"preset": "mid"})
    w.ensure_loaded()
    m = w.model.mlx_model if hasattr(w.model, "mlx_model") else w.model
    dlayers = m.model.layers if hasattr(m, "model") else m.layers
    dkv = {}
    print(f"[probe] dkv: wrapped {_wrap(dlayers, dkv)} attention layers")
    txt = w.generate(prompt=prompt, max_new_tokens=1, temperature=0.0,
                     top_p=1.0, repetition_penalty=1.0)
    _unwrap(dlayers)
    print(f"[probe] dkv tail: {txt[-40:]!r}")

    common = sorted(set(dense) & set(dkv))
    if not common:
        print("[probe] NO COMMON LAYERS — the proxy never recorded on one side. "
              "COVERAGE failure, not a result.")
        return
    print(f"\n{'layer':>6}  {'cos':>9}  {'rel_err':>9}  {'|dense|':>9}  {'|dkv|':>9}")
    import math
    for i in common:
        a, b = dense[i], dkv[i]
        if a.shape != b.shape:
            print(f"{i:>6}  shape {a.shape} vs {b.shape}")
            continue
        na = float(mx.sqrt(mx.sum(a * a)).item())
        nb = float(mx.sqrt(mx.sum(b * b)).item())
        dot = float(mx.sum(a * b).item())
        cos = dot / max(na * nb, 1e-9)
        rel = float(mx.sqrt(mx.sum((a - b) ** 2)).item()) / max(na, 1e-9)
        print(f"{i:>6}  {cos:9.5f}  {rel:9.5f}  {na:9.4f}  {nb:9.4f}")
    print("\n[probe] Compare layer 3 against CUDA's 0.29126 (fail) / 0.27598 (pass).")
    print("        MLX also ~0.28 -> the deviation is inherent to DKV, not the bug.")
    print("        MLX ~0.9+      -> CUDA's decode is far coarser than MLX's, and")
    print("                          THAT gap is what to fix.")


if __name__ == "__main__":
    main()
