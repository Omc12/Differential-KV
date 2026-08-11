"""Repro for intermittent OUTPUT CORRUPTION in the ContinuousBatchEngine path.

STATUS: OPEN. The defect is confirmed real and confirmed to be DKV's, but the
cause is not found. This script exists so the next attempt starts from samples
instead of from scratch.

WHAT IS ESTABLISHED
  * Corruption is real, not a style quibble: outputs contain U+FFFD and mojibake,
    e.g. "- Red: A委员会S���r / - Blue: A委员会Sin / - Green: A委员会Sam".
  * It is DKV's. With DKV_ENGAGE_THRESHOLD=999999 (DKV never engages) the same
    prompt through the same engine was clean 3/3; with DKV engaged it fires at
    roughly 10-20%.
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
    eng = ContinuousBatchEngine(w, max_batch_size=2)
    eng.start()
    bad = 0
    for i in range(N):
        # SESSION_MODE=same reuses ONE session id across every generation, which
        # is what the failing test does; "unique" gives each its own. Session
        # state reuse is where this codebase's previous intermittent corruption
        # lived, so the two rates are a diagnostic, not just a knob.
        _sid = "sess_shared" if os.environ.get("SESSION_MODE", "unique") == "same" else f"sess_{i}"
        q = await eng.submit(_sid, {"prompt": PROMPT, "max_tokens": 128,
                                           "temperature": 0.0, "top_p": 0.9,
                                           "repetition_penalty": 1.15})
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
    await eng.stop()
    print(f"RESULT corrupt={bad}/{N}  mode={os.environ.get('MODE','dkv')}", flush=True)

asyncio.run(main())
