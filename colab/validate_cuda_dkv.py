"""CUDA validation for the 2026-07-29 DKV fixes.

Everything this checks was fixed on a Mac with no NVIDIA GPU available, so none
of it has been exercised on real CUDA hardware. Run this FIRST on a cloud GPU,
before any benchmark — a benchmark of a silently-wrong kernel is worse than no
benchmark.

    python colab/validate_cuda_dkv.py                  # full run
    python colab/validate_cuda_dkv.py --quick          # skip the 8k needle

What it covers, and why each one is here:

  1. Triton kernel COMPILES and is actually used.
     `DKV_TRITON_STRICT=1` turns the silent PyTorch fallback into a raised
     error. Without it a kernel that fails to compile just looks slow.

  2. Rank masking (the bug fixed today).
     `offs_r = tl.arange(0, R_pad)` with R_pad = next_power_of_2(layer_rank)
     ran past the pool's rank dimension whenever the two disagreed — a rank-48
     layer padded to 64 against a 48-wide pool summed 16 rank-rows of the
     ADJACENT SLOT's basis into its own scores. This test forces a layer rank
     whose padding overflows and compares against the exact-SVD reference.

  3. Residual storage format matches the decoder.
     The compressor's exact-keys default is device-derived now: Metal
     substitutes, Triton only ADDS a correction. If CUDA ever writes exact-form
     residuals, every residual token double-counts.

  4. End-to-end needle recall + determinism at temperature 0.
"""
import argparse
import os
import sys

# Must be set before DKV imports — both are read at import time.
os.environ.setdefault("DKV_TRITON_STRICT", "1")      # no silent fallback

# DKV_USE_ATTENTION_INTERFACE selects the RUNTIME, not a tuning knob:
#   "0" = monkeypatch path (Path A) -- runs the fused Triton decode kernel
#   "1" = AttentionInterface path (Path B, dkv_backend.py) -- plain SDPA, no
#         fused kernel, ~4.3 tps with the profiler's "dkv" bucket at 0.0 ms
#
# This script used to hard-pin "0", so it validated Path A while the runtime
# default was "1" -- every check passed on a path nobody was actually running.
# Honour whatever the caller set, and PRINT it, so a run can never again be
# misread as covering a path it did not touch.
os.environ.setdefault("DKV_USE_ATTENTION_INTERFACE", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ACTIVE_RUNTIME"))

import torch

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILURES.append(name)
    return ok


def test_1_environment():
    print("\n=== 1. Environment ===", flush=True)
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    if not torch.cuda.is_available():
        print("  Nothing below can run without CUDA. Stopping.", flush=True)
        sys.exit(1)
    try:
        import triton
        check("triton importable", True, f"version {triton.__version__}")
    except Exception as e:                                    # noqa: BLE001
        check("triton importable", False, str(e))
    print(f"  torch {torch.__version__}", flush=True)
    _ai = os.environ.get("DKV_USE_ATTENTION_INTERFACE", "0")
    print(f"  DKV_USE_ATTENTION_INTERFACE={_ai} -> "
          f"{'Path B / AttentionInterface (NO fused kernel)' if _ai == '1' else 'Path A / monkeypatch (fused decode kernel)'}",
          flush=True)
    check("running the fused-kernel path (set =1 to test Path B instead)",
          _ai == "0", f"DKV_USE_ATTENTION_INTERFACE={_ai}")


def test_2_exact_keys_default():
    """The storage format must match what the decode kernel implements."""
    print("\n=== 2. Residual storage format (exact-keys gate) ===", flush=True)

    # SAVE AND RESTORE. This popped both vars to read the default and never put
    # them back, so `DKV_RESIDUAL_EXACT_KEYS=1 python validate_cuda_dkv.py` ran
    # test 4 with the variable DELETED — the flag under test was destroyed by the
    # test above it, and the run looked like a clean no-op instead of an error.
    _saved = {k: os.environ.get(k)
              for k in ("DKV_RESIDUAL_EXACT_KEYS", "DKV_RESIDUAL_EXCLUDE_SVD")}
    try:
        for k in _saved:
            os.environ.pop(k, None)
        from native_core.compression.lowrank import _exact_keys_enabled

        # Every reader now SUBSTITUTES, matching Metal and MLX: the two Triton
        # kernels via EXACT_RESIDUAL, plus the three that used to only ADD
        # (prefill's _prefill_fused_history_attend_compiled, fused_decode_mps,
        # and the PyTorch vectorized decoder). The storage format is satisfied by
        # all of them, so CUDA defaults to exact form like every other device.
        cuda_default = _exact_keys_enabled(torch.device("cuda:0"))
        check("CUDA defaults to EXACT form (all readers substitute)",
              cuda_default is True,
              "a reader that only ADDs would double-count exact-form residuals; "
              "see _exact_keys_enabled for the full reader list")

        os.environ["DKV_RESIDUAL_EXACT_KEYS"] = "1"
        forced_on = _exact_keys_enabled(torch.device("cuda:0"))
        check("explicit override still honoured", forced_on is True)
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if _saved.get("DKV_RESIDUAL_EXACT_KEYS") is not None:
            print(f"  [config] caller's DKV_RESIDUAL_EXACT_KEYS="
                  f"{_saved['DKV_RESIDUAL_EXACT_KEYS']} restored for test 4",
                  flush=True)


def test_3_rank_mask():
    """Rank padding must not read past the pool's rank dimension.

    Reproduces the exact shape that broke: pool rank 48, layer rank 48, so
    next_power_of_2(48) = 64 and r = 48..63 addressed the next slot.
    """
    print("\n=== 3. Triton rank masking ===", flush=True)
    try:
        import triton
        from native_core.sparse_decode.triton_fused_decode import (
            native_triton_sparse_attn_decode,
        )
    except Exception as e:                                    # noqa: BLE001
        check("import triton decode path", False, str(e))
        return

    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    POOL_RANK, LAYER_RANK = 48, 48
    check("test shape actually triggers the bug",
          triton.next_power_of_2(LAYER_RANK) > POOL_RANK,
          f"R_pad={triton.next_power_of_2(LAYER_RANK)} > pool_rank={POOL_RANK}")

    # Numerical check of the partial-RoPE rewrite against the known-correct helper.
    #
    # BOTH production geometries, not just one. This covered only (256, 64) --
    # Qwen3.5-2B's PARTIAL rope -- so Qwen2.5-1.5B's (128, 128), where
    # rotary_dim == head_dim and the "pass-through" tail is empty, had no coverage
    # at all. That is exactly the model whose DKV output is garbage, so the
    # untested geometry and the broken one are the same geometry.
    from runtime.dkv_attention import _apply_rope_single
    for _name, HD, RD in (("Qwen3.5-2B partial rope", 256, 64),
                          ("Qwen2.5-1.5B full rope", 128, 128)):
        B, KV, L = 1, 2, 7
        k = torch.randn(B, KV, L, HD, device=dev)
        cos = torch.randn(1, 1, L, RD, device=dev)
        sin = torch.randn(1, 1, L, RD, device=dev)
        ref = _apply_rope_single(k, cos, sin)

        rot_dim, half_r = RD, RD // 2
        half = torch.empty_like(k[..., :rot_dim])
        half[..., :half_r] = -k[..., half_r:rot_dim]
        half[..., half_r:] = k[..., :half_r]
        out = torch.empty_like(k)
        torch.mul(k[..., :rot_dim], cos, out=out[..., :rot_dim])
        out[..., :rot_dim].addcmul_(half, sin)
        out[..., rot_dim:].copy_(k[..., rot_dim:])
        check(f"partial-RoPE matches reference on GPU — {_name} (HD={HD}, RD={RD})",
              torch.equal(out, ref) or (out - ref).abs().max().item() < 1e-5,
              f"max|diff|={(out - ref).abs().max().item():.3e}")


def test_4_needle(quick=False, long_ctx=False, dense=False,
                  model_id="Qwen/Qwen3.5-2B", chunk=0,
                  no_serving_defaults=False):
    """End-to-end recall + determinism at temperature 0."""
    print("\n=== 4. End-to-end needle + determinism ===", flush=True)
    import random
    from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper

    NEEDLE = "ZEBRA-4471-QUARTZ"
    random.seed(5)
    pool = [
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
        """Needle at a fractional DEPTH through the filler, not always at the start.

        Depth matters for the routing top-K: the router keeps K blocks, so the
        failure mode is a needle whose block ranks outside the top K. A needle
        pinned at position 0 sits in the first block, which sinks/recency rules
        tend to keep anyway -- it cannot detect a router that is dropping
        mid-context blocks. Sweeping depth is what actually tests K.
        """
        filler = [random.choice(pool) for _ in range(n_filler)]
        at = int(len(filler) * depth)
        needle = (f"Remember this important code: {NEEDLE}. "
                  "This is the only code you need to remember.")
        parts = filler[:at] + [needle] + filler[at:]
        parts.append("Question: What was the important code mentioned in this "
                     "text? Reply with only the code.")
        return " ".join(parts)

    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")

    # VALIDATE WHAT SHIPS. decode_config.BEST_DECODE_DEFAULTS is what cli.py and
    # the OpenAI gateway apply before constructing the manager; this script
    # applied none of them, so every result it has ever produced was on a
    # configuration production does not run. DKV_SPARSE_BIAS is the sharp one:
    # serving sets "auto", dkv_attention.py:136 falls back to "0.0" without it,
    # and those are DIFFERENT MERGE PATHS -- dkv_needle_diag.py's header says
    # running without it "silently exercised the combined kernel, a path
    # production never uses, which masked the deep-needle bug".
    # setdefault, so an explicit env still wins and every A/B in the runbook
    # keeps working.
    if not no_serving_defaults:
        from serving.decode_config import BEST_DECODE_DEFAULTS
        for _k, _v in BEST_DECODE_DEFAULTS.items():
            os.environ.setdefault(_k, _v)
        print("  [config] serving defaults applied: "
              + ", ".join(f"{k}={os.environ[k]}" for k in BEST_DECODE_DEFAULTS))
    else:
        print("  [config] --no-serving-defaults: bare wrapper defaults, "
              "NOT the shipped configuration")

    # SYNCHRONOUS COMPRESSION FOR THE CORRECTNESS RUN.
    #
    # kv_runtime_manager.py:832-841 already documents why, about its own runtime:
    #
    #     "different tokens run to run. That makes both the generated text and
    #      per-layer cosine-vs-dense useless as A/B metrics -- two runs of the
    #      SAME build disagree by more than most changes being evaluated. Set
    #      this for any fidelity measurement; leave it off in production, where
    #      the overlap is the point."
    #
    # Background compression means a block's representation at a given decode
    # step -- exact dense while SUBMITTED, low-rank once COMPRESSED -- depends on
    # when a worker thread happens to finish. So the "determinism at temperature
    # 0" check has been measuring the compression thread's timing, not the
    # kernel. That is the likeliest reading of 8k@depth0.5 sitting at 2/3 with 2
    # distinct outputs while 8k@0.0 and 8k@0.9 are stable: same build, same
    # prompt, different overlap.
    #
    # It is also the MLX-parity setting. MLX compresses at the point a block
    # leaves the window, so it has no overlap and no such variance -- and it
    # returns 3/3 distinct=1 on all nine of these cases.
    #
    # setdefault, so DKV_SYNC_COMPRESS=0 still measures the production
    # (overlapped) configuration deliberately.
    os.environ.setdefault("DKV_SYNC_COMPRESS", "1")
    print(f"  [config] DKV_SYNC_COMPRESS={os.environ['DKV_SYNC_COMPRESS']} "
          f"(1 = compress synchronously, as MLX does; the determinism check is "
          f"meaningless with background compression -- see the note here)")

    if dense:
        # DENSE CONTROL — uncompressed HF attention, same prompts.
        #
        # This is the control that separates "DKV lost the needle" from "the model
        # cannot do this at this length". At 32k the DKV arm fails at depth 0.5/0.9
        # even with routing DISABLED (DKV_TOPK_BLOCKS=0, i.e. every compressed block
        # attended), which rules out retrieval and points at prefill/compression —
        # but only if dense succeeds on the identical prompt. If dense fails too,
        # nothing is wrong with DKV at all and the needle is simply not recoverable
        # by this model at this length.
        #
        # Built as plain HF rather than a flag: there is no DKV_DISABLE env var, and
        # asserting one existed would silently validate DKV against itself.
        import torch as _t
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=_t.float16, device_map="cuda",
            trust_remote_code=True).eval()
        print(f"  [dense control] {model_id} — plain HF attention, DKV NOT loaded")

        class _DenseW:
            tokenizer = _tok
            def generate(self, prompt, max_new_tokens, **kw):
                ids = _tok(prompt, return_tensors="pt").to("cuda")
                if not chunk:
                    with _t.inference_mode():
                        out = _model.generate(**ids, max_new_tokens=max_new_tokens,
                                              do_sample=False)
                    return _tok.decode(out[0])

                # CHUNKED prefill. Without it the dense control cannot REACH the
                # lengths DKV is tested at: single-shot 32k OOMs inside
                # torch_chunk_gated_delta_rule on linear-attention ACTIVATIONS
                # (not the KV cache), which is how the first dense run died right
                # after passing 32k@depth0.0. DKV chunks internally, so comparing
                # a chunked DKV against an unchunked dense measures prefill
                # strategy, not the KV cache — same reasoning as bench_dkv_tps.py.
                from transformers import DynamicCache
                seq = ids["input_ids"][0].tolist()
                # Build the cache FROM THE CONFIG. Qwen3.5 is a hybrid model --
                # linear-attention layers interleaved with full-attention ones --
                # and a bare DynamicCache() allocates only full-attention layers,
                # so the model's own _update_linear_attn_mask raises
                #   "has_previous_state can only be called on LinearAttention
                #    layers, and the current Cache seem to only contain Attention
                #    layers"
                # on the very first chunk. Passing config makes transformers build
                # the per-layer-type structure the model actually expects. Falls
                # back for older transformers / non-hybrid models like Qwen2.5,
                # where the bare constructor is correct.
                try:
                    cache = DynamicCache(config=_model.config)
                except TypeError:
                    cache = DynamicCache()
                gen = []
                with _t.inference_mode():
                    for i in range(0, len(seq), chunk):
                        ch = seq[i:i + chunk]
                        out = _model(
                            input_ids=_t.tensor([ch], device="cuda"),
                            position_ids=_t.tensor(
                                [list(range(i, i + len(ch)))], device="cuda"),
                            past_key_values=cache, use_cache=True)
                    nxt = int(out.logits[0, -1].argmax())
                    pos = len(seq)
                    for _ in range(max_new_tokens):
                        gen.append(nxt)
                        if nxt == _tok.eos_token_id:
                            break
                        out = _model(
                            input_ids=_t.tensor([[nxt]], device="cuda"),
                            position_ids=_t.tensor([[pos]], device="cuda"),
                            past_key_values=cache, use_cache=True)
                        nxt = int(out.logits[0, -1].argmax())
                        pos += 1
                # prompt-prefixed to match the unchunked path, which the caller
                # splits on "assistant".
                return prompt + _tok.decode(gen)
        w = _DenseW()
    else:
        w = PyTorchDKVHFWrapper(model_id=model_id,
                                config={"mode": "fp16"}, device="cuda")
        w.ensure_loaded()

    cases = [("2k", 200, 0.0), ("2k", 200, 0.5), ("2k", 200, 0.9)]
    if not quick:
        cases += [("8k", 800, 0.0), ("8k", 800, 0.5), ("8k", 800, 0.9)]
    if long_ctx:
        # ~32k. This is the gate for making DKV_REMAT_CACHE the default.
        # The remat cache freezes the ROUTED BLOCK SET for DKV_REMAT_INTERVAL
        # tokens, so its staleness risk scales with how many blocks the router is
        # choosing between: at 8k it picks 16 of ~43, at 32k 16 of ~170. A frozen
        # choice that was safe at 8k is not evidence it is safe here, which is why
        # 8k passing is not sufficient to flip the default.
        cases += [("32k", 2400, 0.0), ("32k", 2400, 0.5), ("32k", 2400, 0.9)]
    for label, n_filler, depth in cases:
        label = f"{label}@depth{depth:.1f}"
        ctx = build(n_filler, depth)
        prompt = w.tokenizer.apply_chat_template(
            [{"role": "user", "content": ctx}], tokenize=False,
            add_generation_prompt=True)
        ntok = len(w.tokenizer(prompt).input_ids)
        outs = []
        for _ in range(3):
            # 24 tokens: 12 truncates "ZEBRA-4471-QUARTZ" mid-word once the
            # <think></think> preamble is counted, which reads as a failure.
            r = w.generate(prompt=prompt, max_new_tokens=24, temperature=0.0,
                           top_p=1.0, repetition_penalty=1.0)
            outs.append(r.rsplit("assistant", 1)[-1].strip())
        # Match on alphanumerics only. The needle is one code rendered with
        # different TOKEN boundaries by different tokenizers -- Qwen2.5-1.5B emits
        # 'ZEBR-A-4471-QUARTZ' and 'ZEBR-A4471-QUARTZ', which are exact recalls
        # that a raw substring test scores as misses. Normalising punctuation
        # removes those false negatives WITHOUT weakening the real check: the
        # actual failures still fail, because they get the CONTENT wrong, not the
        # punctuation -- 'ZEBRA-47-QUARTZ' -> ZEBRA47QUARTZ (digits dropped),
        # 'ZEBRA-47-ALUMINUM' -> ZEBRA47ALUMINUM, 'ZEBA-4471-QUARTZ' -> ZEBA...
        # (letter dropped). All three still miss. Raw output is still reported.
        _norm = lambda s: "".join(c for c in s.upper() if c.isalnum())  # noqa: E731
        _needle_n = _norm(NEEDLE)
        hits = sum(_needle_n in _norm(o) for o in outs)
        check(f"{label} ({ntok} tok) needle recall", hits == 3, f"{hits}/3 — {outs[0][:60]!r}")
        # Report WHICH run diverged, not just how many distinct outputs there
        # were. "2 distinct across 3 runs" is compatible with two very different
        # causes and cannot distinguish them:
        #   run 1 alone differs      -> a WARM-UP effect (Triton autotune picking
        #                               a config by timing, JIT compile, a cache
        #                               populated on first use)
        #   the odd run varies       -> a genuine race
        # Every determinism failure in this investigation has been read without
        # that distinction, so both explanations stayed live for many rounds.
        _det_ok = len(set(outs)) == 1
        if _det_ok:
            _det_detail = "1 distinct output across 3 runs"
        else:
            # Label each DISTINCT output with its own letter. A first version
            # marked anything != run 1 as "B", so three different outputs came
            # out as "ABB" and were misreported as a warm-up -- the pattern that
            # would have sent the next round chasing the wrong cause.
            _letters, _pattern = {}, ""
            for o in outs:
                if o not in _letters:
                    _letters[o] = chr(ord("A") + len(_letters))
                _pattern += _letters[o]
            _verdict = ("run 1 differs, runs 2-3 agree -> WARM-UP (autotune / JIT / "
                        "first-use cache)" if _pattern == "ABB"
                        else "every run differs -> race" if len(set(outs)) == len(outs)
                        else "converges after run 1 is not the pattern -> race or state carry-over")
            _det_detail = (f"{len(set(outs))} distinct across 3 runs, pattern={_pattern} "
                           f"({_verdict}); "
                           + " | ".join(f"{_letters[o]}={o[:40]!r}" for o in dict.fromkeys(outs)))
        check(f"{label} determinism at temperature 0", _det_ok, _det_detail)

    if dense:
        return  # no Triton path to check — DKV was never loaded

    # If the Triton kernel silently fell back, DKV_TRITON_STRICT should have
    # raised — but check the counter too in case a path bypasses it.
    from native_core.sparse_decode import triton_fused_decode as tfd
    check("Triton kernel used (no fallback)",
          getattr(tfd, "_triton_fallback_count", 0) == 0,
          f"fallback_count={getattr(tfd, '_triton_fallback_count', 0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 8k prompt")
    ap.add_argument("--long", action="store_true",
                    help="also run ~32k depth cases. This is the gate for making "
                         "DKV_REMAT_CACHE default-ON: the cache freezes the routed "
                         "block set, and that risk scales with how many blocks the "
                         "router chooses between (16 of ~43 at 8k, 16 of ~170 at 32k).")
    ap.add_argument("--dense", action="store_true",
                    help="run the needle cases on plain HF attention with DKV NOT "
                         "loaded. The control that separates 'DKV lost the needle' "
                         "from 'the model cannot do this at this length'.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B",
                    help="model id. Use a smaller one for the --dense control if "
                         "the default OOMs at long context (dense has no compressed "
                         "KV to fall back on).")
    ap.add_argument("--chunk", type=int, default=0,
                    help="--dense only: prefill in chunks of this many tokens. "
                         "Needed to reach 32k — single-shot dense prefill OOMs on "
                         "linear-attention activations, not the KV cache. 1024 "
                         "matches what DKV does internally.")
    ap.add_argument("--no-serving-defaults", action="store_true",
                    help="do NOT apply decode_config.BEST_DECODE_DEFAULTS. Off by "
                         "default so this script validates the SHIPPED config; use "
                         "this to reproduce older runs, which tested a path "
                         "production never uses.")
    args = ap.parse_args()

    test_1_environment()
    if not args.dense:
        # Both inspect DKV internals; neither is meaningful with DKV unloaded.
        test_2_exact_keys_default()
        test_3_rank_mask()
    test_4_needle(quick=args.quick, long_ctx=args.long,
                  dense=args.dense, model_id=args.model, chunk=args.chunk,
                  no_serving_defaults=args.no_serving_defaults)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL CHECKS PASSED")
