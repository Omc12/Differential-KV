#!/usr/bin/env python3
"""Where does the decode time go? — per-preset kernel breakdown.

Runs the SAME setup as run_nat_eval (nat_paper prompt, a chosen preset), prefills,
then decodes N tokens under torch.profiler and reports the top CUDA kernels by
self-time — grouped into MODEL (the eager nf4 GEMM/dequant, shared with dense) vs
DIFFKV (routing + KV reconstruction + the fused sparse-attention kernel) vs other.

This answers, empirically, why tps is what it is: if the nf4-model bucket
dominates, no DiffKV-side change moves tps (that was the 2026-07-18 finding).

Usage (Lightning A100):
    python colab/profile_decode_step.py --model Qwen/Qwen2.5-14B-Instruct --preset low
    python colab/profile_decode_step.py --preset dense   # baseline, no DiffKV
Run it per preset to compare where the time goes.
"""
import os
import sys
import time
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

import torch

PAPER_PATH = os.path.join(ACTIVE, "nat_paper.txt")

# Kernel-name substrings → bucket.  Matched case-insensitively against the op
# name reported by the profiler; first hit wins, else "other".
#
# MODEL: ops that only occur in the Qwen/Llama nf4 forward pass.
_MODEL_HINTS = ("gemm", "cutlass", "dequant", "bitsandbytes", "bnb", "nf4",
                "ampere", "sgemm", "hgemm", "matmul", "linear", "mlp", "silu",
                "rms", "layernorm",
                # Extra nf4 / bitsandbytes CUDA kernel name fragments:
                "gemv_4bit", "kgemm_4bit", "dequantize_blockwise",
                "kDequantizeBlockwise", "volta", "sm80", "sm86")
# DIFFKV: ops that are unambiguously the DiffKV decode path.
# Note: generic aten:: ops (copy_, index, mul, add, cat, gather) are left in
# "other" because they also appear in the model forward pass and cannot be
# reliably attributed without NVTX annotations.  Use --also-dense to see the
# model-only baseline and subtract.
_DIFFKV_HINTS = ("triton", "sparse_attn", "reconstruct", "attend_and_reconstruct",
                 "decode_combined", "fused_decode", "reconstruct_and_score",
                 # Confirmed DiffKV Triton kernel names from triton_fused_decode.py:
                 "_fused_decode_combined", "_triton_sparse_decode",
                 "_dispatch_reduction", "_build_stratified_u",
                 "_gather_routed_blocks")


def _bucket(name: str) -> str:
    n = name.lower()
    for h in _DIFFKV_HINTS:
        if h in n:
            return "diffkv"
    for h in _MODEL_HINTS:
        if h in n:
            return "model"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--preset", default="low",
                    help="low|mid|high, or 'dense' for the no-DiffKV baseline")
    ap.add_argument("--steps", type=int, default=40, help="decode steps to profile")
    ap.add_argument("--warmup", type=int, default=8, help="decode steps before profiling")
    ap.add_argument("--topk", type=int, default=20, help="top ops to print")
    ap.add_argument("--also-dense", action="store_true",
                    help="also run a dense baseline and print side-by-side diff")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This profiler needs CUDA — run it on the A100 box.")
    device = "cuda:0"
    is_dense = (args.preset == "dense")

    from transformers import BitsAndBytesConfig, AutoTokenizer
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(PAPER_PATH, "r", encoding="utf-8") as f:
        paper = f.read()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Provided Text:\n{paper}\n\nSummarize in 100 words."},
    ]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = msgs[1]["content"]
    ids = tok.encode(prompt)
    prompt_len = len(ids)
    print(f"[profile] preset={args.preset} tokens={prompt_len}", flush=True)

    # ── Load + prefill ──
    if is_dense:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quantization_config,
            device_map="auto", trust_remote_code=True).eval()
        _decode = _make_dense_decode(model, ids, device)
    else:
        os.environ["DIFFKV_PRESET"] = args.preset
        os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
        # Same validated defaults as the eval (overridable by the caller's env).
        for _k, _v in (("DIFFKV_COMPRESS_GRAM_SVD", "1"), ("DIFFKV_RANK_BOOST", "off"),
                       ("DIFFKV_RSVD_MAX_RPROJ", "32"), ("DIFFKV_CONTIGUOUS_PREFILL", "1"),
                       ("DIFFKV_CONTIG_UNROTATE", "1")):
            os.environ.setdefault(_k, _v)
        from serving.hf_diffkv_wrapper import DiffKVHFWrapper
        w = DiffKVHFWrapper(model_id=args.model, config={"preset": args.preset,
                            "serving_mode": "balanced"}, torch_dtype=torch.float16,
                            device=device, quantization_config=quantization_config)
        w.ensure_loaded()
        _decode = _make_diffkv_decode(w, ids, device)

    # ── Warm up decode (JIT/allocator) then profile ──
    last = _decode.prefill()
    for _ in range(args.warmup):
        last = _decode.step(last)
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            last = _decode.step(last)
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tps = args.steps / wall
    print(f"\n[profile] {args.steps} steps in {wall:.3f}s → {tps:.1f} tps "
          f"({1000*wall/args.steps:.1f} ms/token)\n", flush=True)

    # ── Aggregate CUDA self-time by op, bucket, and report ──
    evts = prof.key_averages()
    rows = []
    for e in evts:
        cuda_us = getattr(e, "self_cuda_time_total", 0) or getattr(e, "self_device_time_total", 0)
        if cuda_us <= 0:
            continue
        rows.append((e.key, cuda_us / 1000.0, _bucket(e.key)))   # ms
    rows.sort(key=lambda r: -r[1])
    total_ms = sum(r[1] for r in rows) or 1.0
    buckets = {"model": 0.0, "diffkv": 0.0, "other": 0.0}
    for _, ms, b in rows:
        buckets[b] += ms

    print("=" * 78)
    print(f"CUDA time by bucket (self-time over {args.steps} decode steps):")
    for b in ("model", "diffkv", "other"):
        print(f"  {b:<8} {buckets[b]:9.1f} ms total   {100*buckets[b]/total_ms:5.1f}%   "
              f"{buckets[b]/args.steps:6.2f} ms/token")
    print("-" * 78)
    print(f"{'op':<44} {'ms':>9} {'%':>6}  bucket")
    print("-" * 78)
    for name, ms, b in rows[:args.topk]:
        print(f"{name[:44]:<44} {ms:>9.1f} {100*ms/total_ms:>5.1f}%  {b}")
    print("=" * 78)
    print("NOTE: 'other' bucket = generic aten:: ops (copy_, index, cat, gather, mul,\n"
          "      add, neg) that appear in BOTH the model forward and DiffKV block-gather\n"
          "      path. Without NVTX annotations they cannot be attributed. Run with\n"
          "      --also-dense to see the dense baseline and estimate DiffKV overhead:")
    print(f"      python colab/profile_decode_step.py --preset dense --steps {args.steps}")
    print("If 'model' dominates, tps is bound by the eager nf4 forward (shared with"
          "\ndense) and no DiffKV-side change moves it — do the long-context sweep"
          "\ninstead, where KV (and thus the diffkv bucket) becomes the real cost.")


# ── decode drivers ──
class _Driver:
    def __init__(self, prefill_fn, step_fn):
        self.prefill = prefill_fn
        self.step = step_fn


def _make_dense_decode(model, ids, device):
    state = {}
    def prefill():
        with torch.inference_mode():
            out = model(torch.tensor([ids], device=device),
                        position_ids=torch.tensor([list(range(len(ids)))], device=device),
                        use_cache=True)
        state["pkv"] = out.past_key_values
        state["cur"] = len(ids)
        return out.logits[0, -1]
    def step(last_logits):
        nid = int(torch.argmax(last_logits).item())
        with torch.inference_mode():
            out = model(torch.tensor([[nid]], device=device),
                        position_ids=torch.tensor([[state["cur"]]], device=device),
                        past_key_values=state["pkv"], use_cache=True)
        state["pkv"] = out.past_key_values
        state["cur"] += 1
        return out.logits[0, -1]
    return _Driver(prefill, step)


def _make_diffkv_decode(w, ids, device):
    tok, mgr, model = w.tokenizer, w.manager, w.model
    sid = "profile_session"
    state = {}
    def prefill():
        mgr.clear_session(sid)
        if not hasattr(w, "_session_token_ids"):
            w._session_token_ids = {}
        w._session_token_ids[sid] = []
        mgr.init_session(sid, prefill_len=len(ids))
        mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
        model._diffkv_session_ids = [sid]
        CH = int(getattr(getattr(mgr, "config", None), "prefill_chunk_size", 1024))
        if hasattr(mgr, "get_session_micro_block_size"):
            _bc = max(2, int(mgr.get_session_micro_block_size(sid)) + 1)
            CH = ((CH + _bc - 1) // _bc) * _bc
        with torch.inference_mode():
            out = None
            for cs in range(0, len(ids), CH):
                ch = ids[cs:cs+CH]
                out = model(input_ids=torch.tensor([ch], device=device),
                            position_ids=torch.tensor([list(range(cs, cs+len(ch)))], device=device),
                            use_cache=True)
        last = out.logits[0, -1].float().clone()
        if hasattr(mgr, "compress_deferred_prefill_blocks"):
            mgr.compress_deferred_prefill_blocks(sid)
        # drain compression
        _t = time.perf_counter()
        while time.perf_counter() - _t < 30:
            if hasattr(mgr, "finalize_compressed_blocks"):
                mgr.finalize_compressed_blocks()
            sm = getattr(mgr, "_streaming_mgr", None)
            blks = sm.session_blocks.get(sid, {}) if sm else {}
            if sum(1 for L in blks.values() for b in L
                   if getattr(b, "state", None) in ("SUBMITTED", "CPU_COMPRESSED")) == 0:
                break
            time.sleep(0.005)
        if hasattr(mgr, "finalize_srl_index"):
            mgr.finalize_srl_index(sid, cached_len=0)
        state["cur"] = len(ids)
        state["inp"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        state["pos"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        return last
    def step(last_logits):
        nid = int(torch.argmax(last_logits).item())
        state["inp"][0, 0] = nid
        state["pos"][0, 0] = state["cur"]
        with torch.inference_mode():
            out = model(input_ids=state["inp"], position_ids=state["pos"], use_cache=True)
        state["cur"] += 1
        return out.logits[0, -1].float()
    return _Driver(prefill, step)


if __name__ == "__main__":
    main()
