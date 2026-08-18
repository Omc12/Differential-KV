#!/usr/bin/env python3
"""linkbench against the MLX DKV runtime — the metric DKV_ROTATED_POOL is judged on.

WHY THIS AND NOT THE NEEDLE SWEEP
---------------------------------
The needle benchmark plants ONE unique code in bland filler, so it has no
confusable distractors and cannot see the failure the unrotated pool fixes.
linkbench plants N near-identical sentences ("The X Institute is located in Y")
and asks for one of them. Storing POST-RoPE keys bakes in the position a block
held at COMPRESSION time, which is what collapses those distractors together at
long context — the values were never wrong, the positions were.

On CUDA this is worth 40/48 -> 47/48 over 48 seeds, exactly matching dense's
47/48, while every routing knob ever tried moved it nothing.

METHOD, and the parts that are easy to get wrong
------------------------------------------------
* The document builder is IMPORTED from linkbench_cuda.py rather than copied, so
  the two runtimes are graded on token-identical prompts. A divergence in the
  generator would make the comparison meaningless.
* RECORD THE QUESTION MODE NEXT TO EVERY SCORE. linkbench has two, and `chain`
  is the default; `direct` names the intermediate entity outright and is much
  easier. Comparing a `direct` number against a `chain` number reads as a
  regression that is really two different benchmarks — that mistake cost an
  afternoon on the CUDA side. This harness prints the mode in every result line.
* A score is only meaningful next to a CONTROL in the same configuration. Run the
  dense arm too; it is what turns "DKV regressed" into "these are different tasks".
* One seed is not a measurement. Use at least 24.

    SEEDS=$(seq -s, 1 24) QMODE=direct python colab/linkbench_mlx.py
    SEEDS=$(seq -s, 1 24) QMODE=direct ENGINE=dense python colab/linkbench_mlx.py
    SEEDS=$(seq -s, 1 24) QMODE=direct DKV_ROTATED_POOL=0 python colab/linkbench_mlx.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)

# Import the generator, not a copy of it. linkbench_cuda imports torch at module
# scope but does no CUDA work until main(), so this is safe on a Mac.
from colab.linkbench_cuda import build, MAX_NEW, HOPS, SEED     # noqa: E402

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3.5-2B-4bit")
ENGINE = os.environ.get("ENGINE", "dkv")


class MLXDenseWrapper:
    """mlx_lm with DKV not loaded — the control arm."""

    def __init__(self):
        from mlx_lm import generate as mlx_generate, load as mlx_load
        self._model, self._tok = mlx_load(MODEL)
        # mlx_lm hands back a TokenizerWrapper, which is NOT callable — and the
        # shared document builder calls tokenizer(text).input_ids to size the
        # context. Expose the underlying HF tokenizer so both arms are measured
        # with the same builder against the same token counts, and keep the
        # wrapper for generate(), which expects it.
        self.tokenizer = getattr(self._tok, "_tokenizer", self._tok)
        self._gen = mlx_generate

    def generate(self, prompt, max_new_tokens=160, **kw):
        return self._gen(self._model, self._tok, prompt=prompt,
                         max_tokens=max_new_tokens, verbose=False)


def main():
    if ENGINE == "dense":
        w = MLXDenseWrapper()
        print(f"  [dense control] mlx_lm, DKV NOT loaded", flush=True)
    else:
        os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
        from serving.decode_config import BEST_DECODE_DEFAULTS
        for k, v in BEST_DECODE_DEFAULTS.items():
            os.environ.setdefault(k, v)
        cfg = {"preset": os.environ.get("PRESET", "mid")}
        if os.environ.get("RANK"):
            cfg["rank"] = int(os.environ["RANK"])
        if os.environ.get("BLOCK"):
            cfg["micro_block_size"] = int(os.environ["BLOCK"])
            cfg["block_size"] = int(os.environ["BLOCK"])
        from serving.mlx_dkv_wrapper import MLXDKVWrapper
        w = MLXDKVWrapper(model_id=MODEL, config=cfg)
        w.ensure_loaded()
        print(f"  [dkv] preset={cfg['preset']} "
              f"rotated_pool={w.manager.rotated_pool} "
              f"block_size={w.manager.block_size}", flush=True)

    # Tolerate a trailing separator: BSD `seq -s, 1 24` emits one, so the obvious
    # SEEDS=$(seq -s, 1 24) would otherwise die on int('') after the model loaded.
    seeds = [int(x) for x in os.environ.get("SEEDS", str(SEED)).split(",") if x.strip()]
    qmode = os.environ.get("QMODE", "chain")
    hits, ntok = 0, 0
    for i, seed in enumerate(seeds):
        body, question, answer, candidates = build(w.tokenizer, seed)
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}], tokenize=False,
            add_generation_prompt=True)
        ntok = len(w.tokenizer.encode(prompt))
        t0 = time.perf_counter()
        out = w.generate(prompt=prompt, max_new_tokens=MAX_NEW, temperature=0.0,
                         top_p=1.0, repetition_penalty=1.0)
        dt = time.perf_counter() - t0
        if isinstance(out, dict):
            out = out.get("text", str(out))

        # Grade on ATTRIBUTION, not presence: the FIRST candidate name the model
        # emits must be the right one, so echoing the context cannot score.
        ans = out.split(question)[-1] if question in out else out
        # A thinking model spends its budget before answering; say so rather than
        # reporting a recall failure that is really a budget failure.
        truncated = "<think>" in ans and "</think>" not in ans
        low = ans.lower()
        firsts = [(low.find(c.lower()), c) for c in candidates if c.lower() in low]
        firsts = [f for f in firsts if f[0] >= 0]
        said = min(firsts)[1] if firsts else None
        hit = (said is not None and said.lower() == answer.lower())
        hits += bool(hit)
        note = "  TRUNCATED MID-<think>" if (truncated and not hit) else ""
        print(f"  [{i+1}/{len(seeds)}] seed={seed} answer={answer} said={said} "
              f"hit={hit} ({dt:.0f}s){note}", flush=True)

    pool = "n/a" if ENGINE == "dense" else os.environ.get("DKV_ROTATED_POOL", "1")
    print(f"\nLINKBENCH-MLX engine={ENGINE} model={MODEL} ctx={ntok} hops={HOPS} "
          f"qmode={qmode} rotated_pool={pool} "
          f"HITS={hits}/{len(seeds)}", flush=True)


if __name__ == "__main__":
    main()
