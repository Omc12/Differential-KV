"""Repro for intermittent OUTPUT CORRUPTION in the ContinuousBatchEngine path.

STATUS: OPEN. Component NOT established.

DO NOT TRUST UNPATCH=1 AS A CONTROL. It measured 10/10 corrupt and was briefly
read as proving the bug is not DKV's. It proves nothing of the kind: batch_engine
calls self.wrapper.model(input_ids, position_ids, use_cache=True) with NO
past_key_values (:1112, :1136), so DKV's interception is the ONLY thing threading
a KV cache. Removing it leaves the model with no cache at all, and garbage is the
expected result by construction. The flag is kept because it is useful for other
questions, but it cannot separate "engine bug" from "DKV bug".

What IS solid: the engine corrupts (7/10 cold with DKV installed) while
model.generate() on the SAME model in the SAME process is clean 3/3 with coherent
output. So the fault is in the engine-plus-DKV KV path, not in the model or the
environment. Which of the two owns it is still open.

STATUS (rate), NO LONGER INTERMITTENT -- it is 100% DETERMINISTIC and the
repro is one command. That reframing is the main result here; everything below
about "fires ~15%" was an artefact of how it was being sampled.

    REPS=1, fresh process:   ~13/18 CORRUPT across all batches (10/10, then 3/6,
                             then 0/2) -- roughly 70%, NOT the 100% an early
                             10/10 streak suggested. Do not trust a streak.
    REPS=2, fresh process:   generation 0 corrupt, generation 1 CLEAN (3/3)
    PRE-EXISTING:            5/6 at commit 6f24bb19, before any change in this
                             series, so it is not a regression from this work.

So: THE FIRST GENERATION ON A COLD DKV POOL IS USUALLY CORRUPTED (~70%), and every
generation after it in the same process is clean. Sessions are unique per
iteration, so this is not turn-to-turn carry-over; it is cold pool vs warm pool.

Why it ever looked intermittent, twice over:
  * A long-lived process does 1 cold generation and N-1 warm ones, so a REPS=20
    run reports 1/20 and reads as "~5%, flaky".
  * Under pytest every run is a fresh process, hence always cold, hence always
    corrupt -- yet the test only FAILS ~60% of the time, because the assertions
    (newlines / list marker / ASCII punctuation) sometimes hold on garbled text.
    Corruption is 100%; assertion sensitivity is 60%.

THE ROUTED-SET DIFF CANNOT BE DONE AS FRAMED, and that is itself the finding.
DKV_ROUTE_DUMP=1 instruments get_cached_decode_blocks at three depths -- inside
the compressed branch, outside it, and at FUNCTION ENTRY before every early
return. All three print NOTHING on runs that corrupt. So this decode never
consults DKV's block metadata at all: at ~60 prompt tokens against a 257-token
routing block there are no DKV blocks to route, and the corruption happens on the
path taken when the block set is EMPTY. Stop looking at routing/pool/slot code --
none of it runs here -- and look at what dkv_forward does with an empty block set.

THE UNTESTED CONTROL, and the one that matters next: every arm so far has run
through DKVHFWrapper with DKV's attention interception installed, varying only
DKV env flags. The engine has NEVER been run against a completely unpatched
model. Until that is done, "the batch engine corrupts cold generations" and "DKV's
interception corrupts cold generations" are indistinguishable.

Chase the rest as a cold-vs-warm difference, NOT as a race. A cold pool is
freshly torch.zeros'd and lazily allocated; a warm one has slots that have been
written and recycled. Reading a slot that was never written yields zeros, and
zeros here evidently decode to garbage -- which would mean a warm pool MASKS the
same bug with plausible-looking data rather than fixing it. Diff what the routed
set contains on generation 0 versus generation 1.

WHAT IS ESTABLISHED
  * Corruption is real, not a style quibble: outputs contain U+FFFD and mojibake,
    e.g. "- Red: A委员会S���r / - Blue: A委员会Sin / - Green: A委员会Sam".
  * NOT DKV AT ALL -- see UNPATCH=1 above, 10/10 with the interception removed.
  * NOT SHOWN TO BE DKV's COMPRESSION. An earlier note here claimed it was, on
    the strength of DKV_ENGAGE_THRESHOLD=999999 being clean 3/3. That was three
    samples against a ~70% event (P(3 clean) = 0.027) and it did not replicate:
    re-run properly it is 7/10 CORRUPT. Retracted.
    It could not have been what that claim assumed in any case -- the default
    threshold is 4096 (dkv_attention.py:406) and this prompt is ~60 tokens, so
    prefill BYPASSES DKV either way and the two arms are the same code path.
    Whatever this is, it survives DKV being bypassed.
  * It is NOT the streaming detokeniser. batch_engine decodes the FULL generated
    sequence and takes a delta (:1846), never per-token, so U+FFFD is in the
    model's actual output rather than a decode artefact.
  * It is NOT slot recycling: DKV_NO_SLOT_REUSE=1 measured 2/10 against a 2/10
    baseline.
  * It is NOT DKV_DECODE_CACHE: that variable is read ONLY by the MLX wrapper
    (see the note below), so setting it changes nothing on CUDA.
  * It still fires with DKV_COMPRESSED_DECODE=0, so prefill/ingest compression is
    implicated, not only the sparse decode kernel.
  * NOT tiered eviction (DKV_TIER_ENABLED=0): 1/20 vs 1/20, despite maybe_evict's
    own docstring describing exactly this failure ("a routed block could be zeroed
    out from under the launch that was about to read it").
  * NOT session-state reuse: SESSION_MODE=same measured 1/20, same as unique, and
    corruption hits iteration 0 -- a FRESH session's FIRST generation -- so it is
    not accumulation across turns.
  * NOT async SVD publication: DKV_SYNC_COMPRESS=1 measured 1/40 vs 1/40.
  * NOT the async driver: DRIVER=anyio measured 0/20 vs asyncio's 1/20. This had
    been the leading hypothesis (pytest marks coroutine tests pytest.mark.anyio
    and fails far more often) and it is WRONG -- the pytest difference is that
    pytest gives a fresh process, i.e. a cold pool, every run.
  * NOT autotune, and not shape-dependent: VARY_LEN=1 changes the prompt length
    each iteration, so the @triton.autotune key ['N','L_dense'] changes too, and
    it measured 0/14 -- including iteration 0, because VARY_LEN alters the prompt
    even at i=0. Corruption tracks the EXACT prompt on a cold pool, not the
    autotune key.

THE RATE DEPENDS ON THE ASYNC DRIVER, which is the best remaining lead. This
script drives the engine with asyncio.run and sees ~2.5% (1/40). The identical
generation under pytest -- where conftest marks coroutine tests pytest.mark.anyio
-- fails ~60% (2 of 3 runs). Same engine, same model, same prompt, same env; only
the event-loop driver differs. That says the defect is a SCHEDULING-SENSITIVE RACE
between the engine's background generation loop and the consumer awaiting its
queue, not anything in the KV math. Chase it there: run the same body under anyio
in this script and confirm the rate jumps, then look for state the engine touches
between an await point and its next resumption.

THE STRONGEST UNCHASED CLUE. The garbage REPEATS across list items -- the same
wrong fragment ("A委员会S", or "major" in another sample) appears on every line.
That is not random noise; it is one wrong KV entry being attended to repeatedly,
which should be findable by dumping what the routed set contains on a corrupt
step versus a clean one.

MEASURE, DO NOT EYEBALL. The defect fires ~10-20%, so a 3-run or 4-run comparison
proves nothing -- an early 0/4 here looked like a fix and was not. Use REPS >= 20
per arm, and count U+FFFD rather than "no ASCII punctuation" (a model answering in
Chinese trips that innocently).

Many samples of the batch-engine generation, counting UNAMBIGUOUS corruption.

pytest-per-sample costs ~25s and the defect fires ~20% of the time, so proving or
refuting a cause needs more samples than that affords. This runs N generations in
ONE process and counts outputs containing U+FFFD -- a replacement character is
corruption by definition, unlike "no ASCII punctuation", which a model answering
in Chinese trips innocently.
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
