#!/usr/bin/env python3
"""Does DKV_RSVD_MAX_RPROJ=32 change COHERENT GENERATION? Paired, same prompts.

WHY NOT JUST RUN eval_natural_coherence.py TWICE
------------------------------------------------
Two reasons that eval, as shipped, cannot answer this question:

  1. IT PRODUCES NO VERDICT.  It prints generations for a human to read.  Two
     runs give two walls of text and no statement about whether they differ.
  2. ITS PROMPTS ARE TOO SHORT TO ENGAGE DKV AT ALL.  They are a few hundred
     tokens; the dense window covers them, nothing is compressed, and the cap
     -- which only acts during compression -- is a no-op.  Both arms would emit
     byte-identical text and the eval would look like a clean pass while having
     tested NOTHING.  This is the same vacuous-instrument trap the r_proj
     recall harness hit at 4k.

So this script reuses eval_natural_coherence's PROMPTS but (a) forces the dense
window down with DKV_ENGAGE_THRESHOLD so compression actually runs on them, and
(b) runs both arms on the SAME prompt in ONE process and reports an explicit
per-prompt verdict plus the first divergence point.

Forcing the window IS an off-regime stress test, and is meant to be: it puts the
cap on the coherence prompts rather than letting length hide it.  The
production-regime synthesis gate is colab/linkbench_cuda.py at 32k.

USAGE
    python colab/coherence_cap_ab.py
    ENGAGE=128 MODEL=Qwen/Qwen3.5-2B python colab/coherence_cap_ab.py
"""
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

# Force compression to engage on short prompts BEFORE the runtime reads it.
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", os.environ.get("ENGAGE", "128"))
sys.path.insert(0, ACTIVE)
sys.path.insert(0, REPO)

import torch  # noqa: E402
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper  # noqa: E402
from serving.decode_config import BEST_DECODE_DEFAULTS  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "colab"))
from eval_natural_coherence import PROMPTS  # noqa: E402


def main():
    for k, v in BEST_DECODE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    model_id = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
    new_tok = int(os.environ.get("NEW", "220"))

    w = PyTorchDKVHFWrapper(model_id=model_id, config={"mode": "fp16"},
                            device="cuda")
    w.ensure_loaded()

    def gen(prompt, tag):
        w.active_session = tag
        with torch.inference_mode():
            out = w.generate(prompt, max_new_tokens=new_tok, temperature=0.0,
                             top_p=1.0, repetition_penalty=1.0)
        if isinstance(out, dict):
            out = out.get("text", str(out))
        if "<|im_start|>assistant" in out:
            return out.split("<|im_start|>assistant")[-1].strip()
        if "</think>" in out:
            return out.split("</think>")[-1].strip()
        return out[len(prompt):].strip()

    same = 0
    for i, p in enumerate(PROMPTS, 1):
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "system", "content": p["system"]},
             {"role": "user", "content": p["user"]}],
            tokenize=False, add_generation_prompt=True)
        ntok = len(w.tokenizer(prompt).input_ids)

        os.environ["DKV_RSVD_MAX_RPROJ"] = "0"
        a = gen(prompt, "cohere-off-%d" % i)
        os.environ["DKV_RSVD_MAX_RPROJ"] = "32"
        b = gen(prompt, "cohere-32-%d" % i)

        ident = (a == b)
        same += ident
        print("\n" + "=" * 78)
        print("[%d/%d] %s  (prompt %d tok, engage=%s)"
              % (i, len(PROMPTS), p["category"], ntok,
                 os.environ.get("DKV_ENGAGE_THRESHOLD")))
        print("  IDENTICAL" if ident else "  DIFFERS")
        if not ident:
            sm = difflib.SequenceMatcher(None, a, b)
            print("  similarity ratio: %.4f" % sm.ratio())
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "equal":
                    print("  first divergence at char %d:" % i1)
                    print("    cap_off: %r" % a[i1:i1 + 110])
                    print("    cap_32 : %r" % b[j1:j1 + 110])
                    break
        print("-" * 78)
        print("cap_off:\n%s" % a)
        print("-" * 78)
        print("cap_32 :\n%s" % b)

    os.environ.pop("DKV_RSVD_MAX_RPROJ", None)
    print("\n" + "=" * 78)
    print("COHERENCE A/B: %d/%d prompts byte-identical between arms."
          % (same, len(PROMPTS)))
    print("A differing arm is NOT automatically a regression -- a numerical")
    print("change reorders a reduction and can flip a greedy tie.  Read the")
    print("text: the question is whether cap_32 is still coherent, not whether")
    print("it is identical.")


if __name__ == "__main__":
    main()
