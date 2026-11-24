"""Layer-by-layer non-finite hunt on the ContinuousBatchEngine path.

WHY IT EXISTS, and what it ruled OUT -- which is the useful part.

test_formatting fails inside the full suite and passes alone. A logit
fingerprint at the engine's sampling sites (DKV_ENGINE_LOGIT_TRACE=1) showed
'max=nan sum=nan' on the prefill logits, which looked like the answer: greedy
argmax over NaN is implementation-defined, so identical logits could pick
different tokens. The sampler guard that came out of it is a real fix and is
kept (batch_engine._sample now sanitises BEFORE the greedy return).

BUT THE NaN IS NOT THE TEST FAILURE, and the correlation says so plainly:

    pytest, isolation   prefill max=nan        test PASSED
    pytest, full suite  prefill max=3.412109   test FAILED

The NaN run is the one that passed. And after the sampler fix, isolation went
3/3 while the full suite still failed 2/2 -- so sanitising the choice did not
change the outcome, exactly as that table predicts.

WHAT THIS SCRIPT ESTABLISHED. Run standalone it is completely clean and
completely deterministic: 4/4 iterations give identical logits
(argmax=12 max=17.093750 sum=-303740.188), identical text, and NO non-finite
output from any of the 24 decoder layers, the final norm, or lm_head. Forcing
conftest's DKV_COMPRESSED_DECODE=1 / DKV_COMPRESSED_MIN_CTX=8192 changes
nothing -- 4/4 identical again -- so the sparse-vs-dense path choice is not it
either.

So the fault needs the pytest process specifically and does not reproduce in a
plain script, and neither the NaN nor the decode path explains the failure.
Whoever picks this up should start by diffing what pytest's environment does to
the process that a script does not, rather than re-running the layer hunt.

TWO MORE THINGS MEASURED AFTER THE ABOVE, both corrections.

THE ENGINE'S past_kv IS INERT. DKV_ENGINE_LOGIT_TRACE=1 reports the cache length
next to the logits, and it is ZERO on every decode step -- under pytest AND
standalone:

    prefill  cache=none
    decode   cache=0     <- threaded, never accumulates

DKV's attention patch intercepts and serves history from its own session state
rather than populating the past_key_values it was handed. So the engine "owning
a cache" does not do what it says. Confirmed from the other side: reverting
batch_engine.py to before that change leaves the three engine tests at 4 PASSED.
They pass because of the WRAPPER fixes -- the dense-only decode path and the
_MUTATION_OUT_ACTIVE reset -- not because of the cache.

AND THE STANDALONE PATH IS FINE. Same script, no pytest: logits finite
(max=17.09, 16.51, 18.56), text correct, 4/4 identical. Under pytest the same
code gives max=nan. So the NaN needs the pytest process and is not explained by
the cache, the decode path, the pool budget, or the SDPA backend -- all four
tested and eliminated.

Usage: python colab/engine_nan_hunt.py   (env: DKV_MODEL, DKV_ENGINE_LOGIT_TRACE)
"""
import asyncio
import os
import sys

import torch

sys.path.insert(0, r"C:\Users\USER\Desktop\Differential KV\ACTIVE_RUNTIME")
from serving.hf_dkv_wrapper import DKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nProvide a structured list of three major colors. "
    "Use bullet points (with asterisks) and newlines between them. "
    "Write one complete sentence for each color ending with a period.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_seen = []


def _mk_hook(name):
    def _h(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(o):
            return
        if torch.isfinite(o).all():
            return
        i = inp[0] if inp and torch.is_tensor(inp[0]) else None
        in_bad = (i is not None and not torch.isfinite(i).all())
        _seen.append((name, bool(in_bad),
                      int((~torch.isfinite(o)).sum().item()), tuple(o.shape)))
    return _h


async def main():
    os.environ["DKV_ENGINE_LOGIT_TRACE"] = "1"
    w = DKVHFWrapper(MODEL, config={"rank": 16}, device="cuda")
    layers = None
    for path in ("model.layers", "model.model.layers"):
        obj = w
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue
    if layers is None:
        print("could not find decoder layers"); return
    for idx, lyr in enumerate(layers):
        lyr.register_forward_hook(_mk_hook(f"layer{idx}"))
    # The decoder layers were not enough: the final norm and lm_head run AFTER
    # them, so a NaN born there is invisible to layer hooks -- which is exactly
    # what "no layer non-finite, but logits non-finite" would look like.
    for _nm in ("model.norm", "model.model.norm", "lm_head", "model.lm_head"):
        _o = w
        try:
            for _p in _nm.split("."):
                _o = getattr(_o, _p)
            _o.register_forward_hook(_mk_hook(_nm))
        except AttributeError:
            pass
    print(f"hooked {len(layers)} layers + head", flush=True)

    for _it in range(4):
        _seen.clear()
        engine = ContinuousBatchEngine(w, max_batch_size=2)
        engine.start()
        q = await engine.submit(f"sess_{_it}", {
            "prompt": PROMPT, "max_tokens": 16, "temperature": 0.0,
            "top_p": 0.9, "repetition_penalty": 1.15})
        out = []
        while True:
            c = await q.get()
            if "text" in c:
                out.append(c["text"])
            if c.get("is_final"):
                break
        await engine.stop()
        _bad = ", ".join(f"{n}({'in' if b else 'HERE'})" for n, b, _c, _s in _seen[:3])
        print(f"iter {_it}: non_finite_modules={len(_seen)} {_bad} "
              f"| text={''.join(out)[:70]!r}", flush=True)
    return

    print(f"\ntext: {''.join(out)[:120]!r}", flush=True)
    if not _seen:
        print("NO non-finite layer output seen", flush=True)
    else:
        print(f"non-finite layer outputs: {len(_seen)}", flush=True)
        for name, in_bad, n, shape in _seen[:6]:
            src = "INHERITED (input already bad)" if in_bad else "ORIGINATES HERE"
            print(f"  {name}: {n} non-finite of {shape} -- {src}", flush=True)


asyncio.run(main())
