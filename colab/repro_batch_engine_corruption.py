"""RESOLVED (2026-08-18). The ContinuousBatchEngine corruption was a CUDA
STREAM RACE -- two of them, one in decode and one in prefill.

THE BUG. The engine enqueues its forwards on side streams:

    with torch.cuda.stream(decode_stream):     # and prefill_stream
        out = self.wrapper.model(...)

and then reads `out.logits` AFTER the block, on the default stream, which has no
dependency on those streams. No synchronize(), no wait_stream(), no event. So the
read could observe memory the forward had not finished writing.

Reading half-written memory is what all the "corruption" was: logits came back
NaN, greedy argmax over NaN is implementation-defined, and the engine emitted
word salad or coherent-but-different text depending on timing.

THE FIX is one line per stream, ordering the default stream behind the side one:

    torch.cuda.current_stream().wait_stream(decode_stream)
    torch.cuda.current_stream().wait_stream(prefill_stream)

wait_stream rather than synchronize: it orders the streams without blocking the
host.

THE EVIDENCE, and it is exact rather than statistical. Under pytest, before and
after, against the same path run in a bare script:

    step      before            decode fixed      both fixed     standalone
    prefill   max=nan           max=3.408203      17.093750      17.093750
    decode 1  max=nan           max=17.843750     16.515625      16.515625
    decode 2  max=512.000000    max=14.843750     18.562500      18.562500

Both fixes were needed: ordering decode alone left prefill reading half-written
memory, which stopped producing NaN but still produced a DIFFERENT computation
(sum -259.834 against -303740.188, a thousand-fold magnitude difference).

Suite went from 145 passed / 1 failed to 146 passed / 0 failed, twice.

WHY IT TOOK SO LONG, recorded because the wrong turns are instructive. Five
explanations were proposed and eliminated by measurement before this one: SDPA
reduction order (DKV_DETERMINISTIC made it worse), the pool budget (0.5/2.0/8.0
GB identical), the sparse-vs-dense decode path (forcing it changed nothing), the
engine's KV cache (threaded but always length 0 -- DKV intercepts), and the NaN
itself (the NaN run was the one that PASSED). Every one of those is a
CONFIGURATION question, and a race answers to none of them -- which is exactly
why they all measured as no-effect. The thing that found it was tracing the
execution path part by part instead of reasoning about which setting could be
responsible.

--- original notes below, kept because the reasoning is still correct ---
"""

import asyncio, os, sys
ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME")); sys.path.insert(0, ROOT)

N = int(os.environ.get("REPS", "12"))
PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\nProvide a structured list of three major colors. "
          "Use bullet points (with asterisks) and newlines between them. "
          "Write one complete sentence for each color ending with a period.<|im_end|>\n"
          "<|im_start|>assistant\n")


async def main():
    import torch
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    MODEL = os.environ.get("DKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    w = DKVHFWrapper(MODEL, config={"rank": 16}, device="cuda")
    # UNPATCH=1 — THE CONTROL THAT WAS NEVER RUN. Every other arm varies DKV env
    # flags while DKV's attention interception stays installed, so none of them can
    # separate "the batch engine corrupts cold generations" from "DKV's
    # interception does". This removes the interception itself: each attention
    # module goes back to its saved _original_forward (dkv_attention.py:4232), and
    # the lm_head last-token slicing is turned off through the flag that path
    # already honours (_disable_lm_head_slicing, :4238).
    if os.environ.get("UNPATCH") == "1":
        _m = w.model
        _layers = _m.model.layers if hasattr(_m, "model") else _m.layers
        _n = 0
        for _l in _layers:
            _a = getattr(_l, "self_attn", None)
            _of = getattr(_a, "_original_forward", None) if _a is not None else None
            if _of is not None:
                _a.forward = _of
                _n += 1
        _m._disable_lm_head_slicing = True
        print(f"[UNPATCH] restored {_n} attention modules + disabled lm_head slicing",
              flush=True)

    # MINCACHE=1 — THE VALID CONTROL. UNPATCH alone is not one: batch_engine calls
    # model(...) with no past_key_values, so DKV's interception is the only thing
    # threading a KV cache and removing it starves the model by construction.
    # This replaces DKV's forward with a shim that does NOTHING BUT thread a
    # DynamicCache into the original attention. If output is clean, cache
    # threading is fine without DKV and the fault is in DKV's bypass cache
    # handling; if it still corrupts, the engine's own loop is wrong.
    if os.environ.get("MINCACHE") == "1":
        import inspect
        from transformers.cache_utils import DynamicCache
        _cache = DynamicCache()
        _m = w.model
        _layers = _m.model.layers if hasattr(_m, "model") else _m.layers
        _n = 0
        for _l in _layers:
            _a = getattr(_l, "self_attn", None)
            _of = getattr(_a, "_original_forward", None) if _a is not None else None
            if _of is None:
                continue
            try:
                _params = set(inspect.signature(_of).parameters)
            except (ValueError, TypeError):
                _params = {"past_key_value"}

            def _mk(of, params):
                def fwd(hidden_states, **kw):
                    # Same key-selection rule DKV's bypass uses: pass only the
                    # cache kwarg this transformers version actually accepts.
                    if "past_key_value" in params:
                        kw["past_key_value"] = _cache
                    if "past_key_values" in params:
                        kw["past_key_values"] = _cache
                    return of(hidden_states, **kw)
                return fwd

            _a.forward = _mk(_of, _params)
            _n += 1
        _m._disable_lm_head_slicing = True
        print(f"[MINCACHE] {_n} modules -> plain DynamicCache, no DKV logic",
              flush=True)

    eng = ContinuousBatchEngine(w, max_batch_size=2)
    eng.start()
    bad = 0
    for i in range(N):
        # SESSION_MODE=same reuses ONE session id across every generation, which
        # is what the failing test does; "unique" gives each its own. Session
        # state reuse is where this codebase's previous intermittent corruption
        # lived, so the two rates are a diagnostic, not just a knob.
        _sid = "sess_shared" if os.environ.get("SESSION_MODE", "unique") == "same" else f"sess_{i}"
        # VARY_LEN=1 changes the PROMPT LENGTH each iteration. The Triton decode
        # kernels are @triton.autotune'd with key=['N','L_dense'], so a new length
        # is a new autotune key and forces a fresh benchmarking pass. If autotune
        # is what corrupts the first generation, corruption should follow the
        # SHAPE CHANGES rather than sitting only on iteration 0.
        # Tag every ROUTE-DUMP line with which generation produced it, so gen 0
        # (cold pool) and gen 1 (warm) can be separated in one capture.
        try:
            from native_core import kv_runtime_manager as _krm
            _krm._ROUTE_DUMP_STATE["gen"] = i
        except Exception:
            pass
        _p = PROMPT
        if os.environ.get("VARY_LEN") == "1":
            _p = PROMPT.replace("three major colors",
                                "three major colors " + ("and shades " * (i % 7)))
        q = await eng.submit(_sid, {"prompt": _p, "max_tokens": 128,
                                           "temperature": 0.0, "top_p": 0.9,
                                           "repetition_penalty": float(os.environ.get("REP_PEN","1.15"))})
        buf = []
        while True:
            c = await asyncio.wait_for(q.get(), timeout=60.0)
            if c.get("error"):
                break
            buf.append(c.get("text", ""))
            if c.get("is_final"):
                break
        txt = "".join(buf)
        corrupt = "\ufffd" in txt
        bad += int(corrupt)
        if corrupt:
            print(f"  [{i}] CORRUPT: {txt[:90]!r}", flush=True)
    # DIRECT=1 — after the engine run, generate the SAME prompt straight through
    # model.generate() in the SAME process. If the engine's output is corrupt
    # while the direct one is clean, the defect is in the engine's prefill/decode
    # loop rather than the model or the environment, and that is the whole
    # remaining question.
    if os.environ.get("DIRECT") == "1":
        import torch as _t
        _tok = w.tokenizer
        _ids = _tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
        with _t.no_grad():
            _out = w.model.generate(_ids, max_new_tokens=128, do_sample=False,
                                    pad_token_id=_tok.eos_token_id)
        _txt = _tok.decode(_out[0][_ids.shape[1]:], skip_special_tokens=True)
        print(f"DIRECT corrupt={'�' in _txt}  text={_txt[:110]!r}", flush=True)

    await eng.stop()
    print(f"RESULT corrupt={bad}/{N}  mode={os.environ.get('MODE','dkv')}", flush=True)

# DRIVER=anyio runs the SAME coroutine through anyio instead of asyncio.run.
# This is the experiment the header calls for: under pytest (which conftest marks
# pytest.mark.anyio) this defect fires ~60%, while asyncio.run sees ~2.5%. If the
# driver is what moves the rate, the bug is a scheduling-sensitive race in the
# engine rather than anything in the KV math, and that is a completely different
# place to look.
_driver = os.environ.get("DRIVER", "asyncio")
if _driver == "anyio":
    import anyio
    anyio.run(main)
else:
    asyncio.run(main())
