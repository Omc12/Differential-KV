"""Throughput through the REAL generate() path, with proof of which decode path ran.

Why this exists rather than profile_decode_step.py: that harness drives
`model(input_ids=...)` directly, bypassing the wrapper's session setup and
routing. DKV's fused decode never engages there, so it measures DKV's
bookkeeping with the kernel switched off -- 4.3 tps, with the profiler's "dkv"
bucket at 0.0 ms and no "COMBINED path ACTIVE" banner. Real users go through
generate(), which is what validate_cuda_dkv.py exercised and where the fused
path DID engage.

Every number below is paired with evidence of the path taken, so a fast or slow
result cannot be misread as the other path's.

    python colab/bench_dkv_tps.py                      # DKV, mid preset
    python colab/bench_dkv_tps.py --dense              # baseline, DKV off
    python colab/bench_dkv_tps.py --ctx 32000          # long-context sweep point
"""
import argparse
import os
import sys
import time

os.environ.setdefault("DKV_TRITON_STRICT", "1")   # a broken kernel must raise, not creep

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "ACTIVE_RUNTIME"))

import torch


def build_prompt(tok, target_tokens):
    import random
    random.seed(5)
    pool = [
        "The morning fog rolled over the hills before the sun broke through the clouds.",
        "Researchers published a new dataset covering climate trends across five continents.",
        "The old library smelled of dust and aging paper, a comfort to regular visitors.",
        "Markets fluctuated throughout the week as investors weighed new economic data.",
        "A gentle breeze carried the scent of pine through the quiet mountain trail.",
        "The committee reviewed dozens of proposals before selecting a final design.",
    ]
    parts, n = [], 0
    while n < target_tokens:
        s = random.choice(pool)
        parts.append(s)
        n += len(tok(s).input_ids)
    parts.append("Summarise the material above in one sentence.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--preset", default="mid")
    ap.add_argument("--ctx", type=int, default=13000, help="approx prompt tokens")
    ap.add_argument("--steps", type=int, default=64, help="decode tokens to time")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--dense", action="store_true", help="disable DKV for a baseline")
    ap.add_argument("--chunk", type=int, default=0,
                    help="dense baseline only: prefill in chunks of this many "
                         "tokens (DKV uses 1024 on CUDA). 0 = HF single-shot.")
    args = ap.parse_args()

    os.environ.setdefault("DKV_PRESET", args.preset)

    # Dense baseline loads a PLAIN HF model with no DKV interception at all --
    # the same thing profile_decode_step.py does for --preset dense. There is no
    # DKV_DISABLE env var; asserting one existed would have silently benchmarked
    # DKV against itself.
    if args.dense:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="cuda",
            trust_remote_code=True).eval()

        class _W:
            tokenizer = tok
            def generate(self, prompt, max_new_tokens, **kw):
                ids = tok(prompt, return_tensors="pt").to("cuda")
                if not args.chunk:
                    with torch.inference_mode():
                        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                                             do_sample=False)
                    return tok.decode(out[0])

                # CHUNKED dense prefill, matching what DKV does.
                #
                # Without this the comparison is unfair in DKV's favour: HF
                # prefills all N tokens in ONE forward, so at 63k it OOMed inside
                # torch_chunk_gated_delta_rule (linear-attention ACTIVATIONS,
                # not the KV cache) while DKV -- which chunks at 1024 -- survived.
                # That reads as a DKV reach win but is really a prefill-strategy
                # difference. This makes both sides chunk so any remaining gap is
                # attributable to the KV cache itself.
                from transformers import DynamicCache
                seq = ids["input_ids"][0].tolist()
                cache = DynamicCache()
                with torch.inference_mode():
                    for i in range(0, len(seq), args.chunk):
                        ch = seq[i:i + args.chunk]
                        out = model(
                            input_ids=torch.tensor([ch], device="cuda"),
                            position_ids=torch.tensor(
                                [list(range(i, i + len(ch)))], device="cuda"),
                            past_key_values=cache, use_cache=True)
                    nxt = int(out.logits[0, -1].argmax())
                    pos = len(seq)
                    for _ in range(max_new_tokens):
                        out = model(
                            input_ids=torch.tensor([[nxt]], device="cuda"),
                            position_ids=torch.tensor([[pos]], device="cuda"),
                            past_key_values=cache, use_cache=True)
                        nxt = int(out.logits[0, -1].argmax())
                        pos += 1
                return ""
        w = _W()
    else:
        from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
        w = PyTorchDKVHFWrapper(model_id=args.model, config={"mode": "fp16"},
                                device="cuda")
        w.ensure_loaded()
    prompt_body = build_prompt(w.tokenizer, args.ctx)
    prompt = w.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_body}],
        tokenize=False, add_generation_prompt=True)
    ntok = len(w.tokenizer(prompt).input_ids)

    # Warmup: JIT/compile/allocator out of the way.
    w.generate(prompt=prompt, max_new_tokens=args.warmup, temperature=0.0,
               top_p=1.0, repetition_penalty=1.0)

    # TWO-POINT TIMING -- required, not a refinement.
    #
    # generate() RE-PREFILLS the whole prompt on every call, so timing a single
    # generate(N) and dividing by N reports (prefill + N*decode)/N, not decode
    # throughput. At 63k the prefill swamps 64 decode tokens entirely; the
    # earlier "2.8 tps" was mostly amortised prefill wearing a decode label.
    #
    #   t(N)  = prefill + N*d
    #   t(2N) = prefill + 2N*d
    #   => d       = (t(2N) - t(N)) / N
    #   => prefill =  t(N) - N*d
    #
    # Backend-agnostic, so DKV and the dense baseline are measured identically.
    def timed(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0,
                   top_p=1.0, repetition_penalty=1.0)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    N = args.steps
    t1, t2 = timed(N), timed(2 * N)
    d_s = max((t2 - t1) / N, 1e-9)          # seconds per decoded token
    prefill_s = max(t1 - N * d_s, 0.0)

    mode = f"DENSE (DKV off{', chunked prefill ' + str(args.chunk) if args.chunk else ', single-shot prefill'})" if args.dense else f"DKV preset={args.preset}"
    print("\n" + "=" * 66)
    print(f"  {mode}   model={args.model}")
    print(f"  prompt {ntok} tok   ({t1:.2f}s for {N} tok, {t2:.2f}s for {2*N} tok)")
    print(f"  PREFILL {prefill_s:7.2f} s   ({1000*prefill_s/max(ntok,1):.3f} ms/prompt-token)")
    print(f"  DECODE  {1/d_s:7.1f} tps ({1000*d_s:.1f} ms/token)   <-- prefill excluded")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    # Evidence of which decode path actually ran -- without this the number is
    # uninterpretable, which is the mistake that produced the 4.3 tps figure.
    if not args.dense:
        try:
            from native_core.sparse_decode import triton_fused_decode as tfd
            fb = getattr(tfd, "_triton_fallback_count", None)
            print(f"  triton fallback count: {fb}"
                  + ("   <-- 0 means the fused kernel ran" if fb == 0 else
                     "   <-- NONZERO: ran the slow PyTorch fallback"))
        except Exception as e:                                # noqa: BLE001
            print(f"  (could not read fallback counter: {e})")
    print("=" * 66)


if __name__ == "__main__":
    main()
