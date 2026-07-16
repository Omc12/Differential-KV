#!/usr/bin/env python3
"""Neighborhood Attention Transformer Paper Evaluation Script.

This script runs comparative prompt evaluations of standard Dense attention against
various DiffKV (Differential KV) configurations (presets, early rank boost, factual store)
using Qwen/Qwen2.5-14B-Instruct in 4-bit weights.
"""

import os
import sys
import ssl
import urllib3
import requests
from urllib3.exceptions import InsecureRequestWarning

# Global absolute SSL verification bypass for all python libraries (including urllib3, requests, HF)
urllib3.disable_warnings(InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey-patch urllib3 to always use unverified SSL context
import urllib3.util.ssl_
urllib3.util.ssl_.create_urllib3_context = lambda *args, **kwargs: ssl._create_unverified_context()

# Monkey-patch requests
old_merge_settings = requests.Session.merge_environment_settings
def patched_merge_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings
requests.Session.merge_environment_settings = patched_merge_settings

# Prevent PyTorch VRAM fragmentation OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Diagnostics are opt-in.  The old unconditional setting printed every block
# decision during 4K–128K sweeps and added noticeable host/I/O overhead.
os.environ.setdefault("DIFFKV_DIAG", "0")

# Ensure active runtime path and C++ compiled library directory are in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
CORE_DIR = os.path.join(ACTIVE, "native_core", "diffkv_core")

if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import json
import time
import argparse
import gc
import re
import math
import subprocess
import torch

# Define Prompts
PROMPT1_TEXT = """Use only the supplied text.

Explain why Neighborhood Attention is designed around each token's nearest neighbors rather than fixed non-overlapping windows.

Your answer must explain the reasoning chain:
- the boundary behavior of zero-padded local attention,
- the restriction created by non-overlapping window partitions,
- how per-token neighborhoods change the attention span,
- what Neighborhood Attention approaches as the neighborhood size reaches its maximum,
- and why shifted windows are needed in one design but not as a manual operation in Neighborhood Attention.

Do not summarize the entire work.
Do not discuss benchmark accuracy, datasets, CUDA, or implementation speed.
Do not introduce mechanisms not explicitly stated in the supplied text.

Write one coherent technical explanation of 120 to 180 words."""

PROMPT2_TEXT = """Use only the supplied text.

A researcher claims:
"Neighborhood Attention is simply Window Self-Attention with overlapping windows."

Evaluate this claim.

Identify which parts of the claim are superficially plausible and then explain precisely why the definition of Neighborhood Attention is different.

Your explanation must distinguish the attention span of an individual token from the partitioning of the entire feature map.

Do not use outside knowledge.
Do not discuss experimental results.
Maximum 180 words."""

PAPER_PATH = os.path.join(ACTIVE, "nat_paper.txt")

def build_prompt(tokenizer, paper_content, prompt_instructions):
    """Build a prompt using the model's own chat template.

    Works for any HuggingFace model — Qwen (ChatML), Llama 3 (llama3), Mistral,
    Phi-3, Gemma, etc.  Falls back to a plain concatenation if the tokenizer has
    no chat template defined.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's request strictly using the provided context.",
        },
        {
            "role": "user",
            "content": f"Provided Text:\n{paper_content}\n\nInstructions:\n{prompt_instructions}",
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: plain concatenation (still correct for most models that at least tokenize
        # system+user text without special tokens).
        system = messages[0]["content"]
        user   = messages[1]["content"]
        return f"{system}\n\n{user}\n\nAssistant:"


def analytic_kv_bytes(mgr, seq_len, sid):
    """Calculate the footprint of the DiffKV Cache."""
    L = mgr.num_layers
    Hkv = mgr.kv_heads
    d = mgr.head_dim
    B = mgr.block_size
    r = mgr.rank
    M = mgr.max_blocks
    Dmax = mgr.max_dense_len
    R = mgr.max_residual
    fp16 = 2
    
    kv_tok = Hkv * d * fp16 * 2
    lowrank_block = ((B - 1) * r * fp16
                     + 2 * Hkv * r * d * fp16
                     + 2 * Hkv * d * fp16
                     + 2 * Hkv * d * fp16
                     + 8)
    residual_block_max = R * kv_tok
    per_block = lowrank_block + residual_block_max
    dense_alloc = Dmax * kv_tok
    
    s0 = mgr.sessions.get(sid)
    nb = s0["num_blocks"][0] if s0 else 0
    dl = s0["dense_lens"][0] if s0 else 0
    res_n0 = s0["comp_res_n"][0][:nb] if s0 else []
    res_tokens_used = int(sum(res_n0))
    
    store_used = L * (nb * lowrank_block + res_tokens_used * kv_tok + dl * kv_tok)
    return {
        "store_used_bytes": store_used,
    }


def wait_for_compression(mgr, session_id: str) -> bool:
    """Drain prefill compression before starting the first decode step."""
    streaming_mgr = getattr(mgr, "_streaming_mgr", None)
    if streaming_mgr is None:
        return True
    timeout_s = float(os.environ.get("DIFFKV_COMPRESSION_TIMEOUT_S", "30"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hasattr(mgr, "finalize_compressed_blocks"):
            mgr.finalize_compressed_blocks()
        blocks = streaming_mgr.session_blocks.get(session_id, {})
        pending = sum(
            1 for layer_blocks in blocks.values()
            for block in layer_blocks
            if getattr(block, "state", None) in ("SUBMITTED", "CPU_COMPRESSED")
        )
        if pending == 0:
            return True
        time.sleep(0.002)
    print(f"[NAT eval] WARNING: compression barrier timed out for {session_id}", flush=True)
    return False

def run_worker(config_name, model_id):
    os.environ["DIFFKV_FACTUAL_STORE"] = "0"
    os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "0"

    is_compressed = (config_name != "dense")
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if is_compressed else "0"

    if is_compressed:
        if config_name == "low_preset":
            os.environ["DIFFKV_PRESET"] = "low"
        elif config_name == "mid_preset":
            os.environ["DIFFKV_PRESET"] = "mid"
        elif config_name == "high_preset":
            os.environ["DIFFKV_PRESET"] = "high"
        elif config_name == "early_boost":
            os.environ["DIFFKV_PRESET"] = "mid"
            os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "1"
        elif config_name == "factual_store":
            os.environ["DIFFKV_PRESET"] = "mid"
            os.environ["DIFFKV_FACTUAL_STORE"] = "1"
        elif config_name == "combined":
            os.environ["DIFFKV_PRESET"] = "mid"
            os.environ["DIFFKV_EARLY_LAYER_RANK_BOOST"] = "1"
            os.environ["DIFFKV_FACTUAL_STORE"] = "1"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if not os.path.exists(PAPER_PATH):
        raise FileNotFoundError(f"Context paper file not found at {PAPER_PATH}")

    with open(PAPER_PATH, "r", encoding="utf-8") as f:
        paper_text = f.read()

    results = {}

    from transformers import BitsAndBytesConfig, AutoTokenizer
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load tokenizer ONCE (before the prompts loop) so we can use
    #    apply_chat_template for model-agnostic prompt formatting.
    print(f"[NAT eval] Loading tokenizer for {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Build all prompts using the model's own chat template (works for Qwen,
    # Llama 3, Mistral, Phi-3, Gemma, etc. — no hardcoded special tokens).
    all_prompts = [
        (idx, build_prompt(tok, paper_text, instr))
        for idx, instr in enumerate([PROMPT1_TEXT, PROMPT2_TEXT], 1)
    ]

    # Derive stop token IDs from the tokenizer (model-agnostic).
    _stop_ids = set()
    for _sid in [tok.eos_token_id, tok.pad_token_id]:
        if _sid is not None:
            _stop_ids.add(_sid)
    # Also add any EOS-like special tokens that are registered in the vocab.
    for _word in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>", "<|endoftext|>"]:
        _tid = tok.convert_tokens_to_ids(_word)
        if _tid is not None and _tid != tok.unk_token_id:
            _stop_ids.add(_tid)

    # Prefill chunk size.  The DiffKV branch overwrites this from the active
    # preset and then rounds it up to the block capacity; the dense branch used
    # to keep the 128 default, so dense ran ~105 forwards over a 13K prompt
    # while DiffKV ran ~13.  That is a per-forward-overhead difference, not an
    # attention difference, and it silently inflated the dense prefill baseline.
    # Both branches now start from the same value.
    CH = int(os.environ.get("DIFFKV_PREFILL_CHUNK_SIZE", "1024"))

    with torch.inference_mode():
        if is_compressed:
            # ── Load DiffKV wrapper ONCE, iterate over all prompts ──────────
            from serving.hf_diffkv_wrapper import DiffKVHFWrapper
            cfg = {
                "preset": os.environ.get("DIFFKV_PRESET", "mid"),
                "serving_mode": "balanced",
            }
            if config_name in ["early_boost", "combined"]:
                cfg["early_layer_rank_boost"] = True
            if config_name in ["factual_store", "combined"]:
                cfg["factual_store"] = True

            w = DiffKVHFWrapper(
                model_id=model_id,
                config=cfg,
                torch_dtype=torch.float16,
                device=device,
                quantization_config=quantization_config,
            )
            w.ensure_loaded()
            tok, mgr, model = w.tokenizer, w.manager, w.model
            # Use the wrapper's stop token set (superset of what we derived above)
            stop_ids = getattr(w, "stop_token_ids", _stop_ids) | _stop_ids
            # Start from the configured outer prefill chunk size.  It is rounded
            # to the active block capacity after each session is initialized.
            _cfg = getattr(mgr, "config", None)
            if _cfg is not None:
                CH = _cfg.prefill_chunk_size
            print(f"[NAT eval] Outer prefill chunk size: CH={CH}", flush=True)
            print("[NAT eval] CUDA exact-prefill mode: SVD deferred until boundary; contiguous block layout", flush=True)
            # CH controls forward-pass throughput, not compression.
            # Compression runs post-forward (compress_deferred_prefill_blocks) at any CH.
            for idx, full_prompt in all_prompts:
                ids = tok.encode(full_prompt)
                prompt_len = len(ids)

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                sid = f"prompt_{idx}"
                mgr.clear_session(sid)
                if not hasattr(w, "_session_token_ids"):
                    w._session_token_ids = {}
                w._session_token_ids[sid] = []

                mgr.init_session(sid, prefill_len=prompt_len)
                mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
                model._diffkv_session_ids = [sid]

                if torch.cuda.is_available() and hasattr(mgr, "get_session_micro_block_size"):
                    _mbs = mgr.get_session_micro_block_size(sid)
                    _block_capacity = max(2, int(_mbs) + 1)
                    _base_chunk = int(getattr(_cfg, "prefill_chunk_size", CH))
                    CH = ((_base_chunk + _block_capacity - 1) // _block_capacity) * _block_capacity
                    print(f"[NAT eval] Aligned CUDA chunk size: CH={CH} (block_capacity={_block_capacity})", flush=True)

                t_prefill_start = time.perf_counter()
                for cs in range(0, len(ids), CH):
                    ch = ids[cs:cs+CH]
                    out = model(
                        input_ids=torch.tensor([ch], device=device),
                        # The second positional argument of a HF causal-LM
                        # forward is attention_mask, not position_ids.  Passing
                        # this position tensor positionally silently made every
                        # outer CUDA chunk restart RoPE at position zero.
                        position_ids=torch.tensor(
                            [list(range(cs, cs + len(ch)))], device=device
                        ),
                        use_cache=True,
                    )
                    # Do not publish lossy SVD blocks between prefill chunks.
                    # CUDA's chunked prefill reads the previous chunks back through
                    # the same block manager; compressing here would make the next
                    # chunk attend reconstructed KV and can change the first decode
                    # token (including an immediate EOS).  MLX keeps raw prefill KV
                    # available through the whole prefill and only switches to the
                    # compressed store at the prefill->decode boundary.

                # The forward passes are done; everything after this point is
                # DiffKV-specific cache construction.  Time it separately —
                # folding it into prefill_time made a 17s number that is really
                # "forward + SVD + SRL index" look like a like-for-like
                # comparison against dense's forward-only prefill.
                # Measured before the diagnostic below so that the diagnostic's
                # own .item()/.tolist() syncs land outside the timer, matching
                # where the dense branch stops its clock.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_time = time.perf_counter() - t_prefill_start
                prefill_forward_vram = (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0
                )

                # Snapshot the final prefill logits before compression/SRL
                # finalization can allocate or mutate CUDA workspaces.
                last_logits_gpu = out.logits[0, -1].float().clone()
                _prefill_topv, _prefill_topi = torch.topk(last_logits_gpu, k=5)
                _prefill_first_id = int(_prefill_topi[0].item())
                print(
                    f"[DIAG] prefill-next prompt{idx}: first_id={_prefill_first_id} "
                    f"stop={_prefill_first_id in stop_ids} "
                    f"top5={[int(x) for x in _prefill_topi.tolist()]}",
                    flush=True,
                )

                t_compress_start = time.perf_counter()
                # Compression is intentionally started once, after all prefill
                # forwards have completed, so validation measures exact causal
                # prefill rather than a lossy mid-prefill approximation.
                if hasattr(mgr, "compress_deferred_prefill_blocks"):
                    mgr.compress_deferred_prefill_blocks(sid)
                wait_for_compression(mgr, sid)
                if hasattr(mgr, "finalize_srl_index"):
                    mgr.finalize_srl_index(sid, cached_len=0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                compress_time = time.perf_counter() - t_compress_start
                prefill_compress_vram = (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0
                )

                # ── Block-state diagnostic (one-time, remove after fix) ──
                _smgr = getattr(mgr, "_streaming_mgr", None)
                if _smgr is not None:
                    _blks = _smgr.session_blocks.get(sid, {}).get(0, [])
                    from collections import Counter
                    _states = Counter(b.state for b in _blks)
                    print(f"[DIAG] Layer-0 block states after prefill: {dict(_states)}", flush=True)
                    print(f"[DIAG] Total layer-0 blocks: {len(_blks)}", flush=True)

                    # Compute total_seq_len as compress_deferred_blocks would see it
                    if _blks:
                        _lb = _blks[-1]
                        _tsl = _lb.anchor_idx + _lb.token_count()
                        print(f"[DIAG] total_seq_len (last blk anchor={_lb.anchor_idx} + tcount={_lb.token_count()}) = {_tsl}", flush=True)
                        print(f"[DIAG] recency_window={_smgr.recency_window}  threshold={_tsl - _smgr.recency_window}", flush=True)

                    # Per-block eligibility dump for first 10 blocks
                    from native_core.streaming_sparse_ingest import _is_block_compression_eligible
                    _tsl2 = _blks[-1].anchor_idx + _blks[-1].token_count() if _blks else 0
                    for _i, _b in enumerate(_blks[:10]):
                        _elig = _is_block_compression_eligible(_b, is_last_block=(_i == len(_blks) - 1))
                        _wok = (_b.anchor_idx + _b.token_count()) < (_tsl2 - _smgr.recency_window)
                        _ak_shape = tuple(_b.active_k.shape) if _b.active_k is not None else None
                        print(f"[DIAG] blk#{_i} anchor={_b.anchor_idx} tcount={_b.token_count()} "
                              f"state={_b.state} eligible={_elig} window_ok={_wok} "
                              f"mbs={_b.micro_block_size} ak={_ak_shape}", flush=True)
                # ── End diagnostic ──

                # Excludes the block-state diagnostic dump above, which is
                # measurement scaffolding rather than runtime work.
                prefill_time = forward_time + compress_time

                peak_prefill_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                cur = prompt_len
                gen_ids = []
                # Pre-allocate static decode tensors — no per-step allocation
                _inp = torch.zeros((1, 1), dtype=torch.long, device=device)
                _pos = torch.zeros((1, 1), dtype=torch.long, device=device)
                t_decode_start = time.perf_counter()
                for _ in range(256):
                    nid_gpu = torch.argmax(last_logits_gpu)
                    nid = int(nid_gpu.item())  # one scalar D2H — unavoidable for stop check
                    if nid in stop_ids:
                        break
                    gen_ids.append(nid)
                    # The prompt token IDs were registered before prefill and
                    # SRL is finalized before decode.  Appending every decoded
                    # token here would repeatedly torch.cat the full 13K-token
                    # prompt on the CPU and is not needed for this benchmark.
                    _inp[0, 0] = nid
                    _pos[0, 0] = cur
                    # Keep position_ids explicit here as well.  Passing _pos as
                    # the second positional argument makes it attention_mask and
                    # can route the single-token step through the wrong cache
                    # semantics.  Explicit use_cache is required for the CUDA
                    # sparse decode branch.
                    out = model(
                        input_ids=_inp,
                        position_ids=_pos,
                        use_cache=True,
                    )
                    last_logits_gpu = out.logits[0, -1].float()
                    cur += 1
                if torch.cuda.is_available():
                    # Include the final queued CUDA work in the measurement.
                    # Without this, a run that reaches the token cap can report
                    # an artificially high TPS because the last forward is still
                    # asynchronous when the timer stops.
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - t_decode_start
                peak_decode_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

                generated_text = tok.decode(gen_ids)
                kv = analytic_kv_bytes(mgr, prompt_len, sid)
                kv_vram = kv.get("store_used_bytes", 0) / 1e9

                results[f"prompt{idx}"] = {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "generated_tokens": len(gen_ids),
                    "prefill_time_s": prefill_time,
                    "prefill_forward_s": forward_time,
                    "prefill_compress_s": compress_time,
                    "prefill_chunk_size": CH,
                    "decode_time_s": decode_time,
                    "decode_tps": len(gen_ids) / decode_time if decode_time > 0 else 0.0,
                    "peak_prefill_vram_gb": peak_prefill_vram,
                    "prefill_forward_vram_gb": prefill_forward_vram,
                    "prefill_compress_vram_gb": prefill_compress_vram,
                    "peak_decode_vram_gb": peak_decode_vram,
                    "pool_physical_mb": (
                        mgr.native_pool._pool_mb()
                        if getattr(mgr, "native_pool", None) is not None
                        and hasattr(mgr.native_pool, "_pool_mb")
                        else 0.0
                    ),
                    "kv_cache_vram_gb": kv_vram,
                    "output_text": generated_text,
                }

            try:
                w.close()
            except Exception:
                pass
            del w, model, mgr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            # ── Dense baseline — load model once for all prompts ──────────
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()

            for idx, full_prompt in all_prompts:
                ids = tok.encode(full_prompt)
                prompt_len = len(ids)

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t_prefill_start = time.perf_counter()
                past_key_values = None
                for cs in range(0, len(ids), CH):
                    ch = ids[cs:cs+CH]
                    pos = torch.tensor([list(range(cs, cs+len(ch)))], device=device)
                    out = model(torch.tensor([ch], device=device), position_ids=pos, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # Dense has no compression stage; its prefill is all forward.
                forward_time = time.perf_counter() - t_prefill_start
                compress_time = 0.0
                # Keep logits on GPU — no D2H sync during prefill
                last_logits_gpu = out.logits[0, -1].float()
                _prefill_topv, _prefill_topi = torch.topk(last_logits_gpu, k=5)
                _prefill_first_id = int(_prefill_topi[0].item())
                print(
                    f"[DIAG] dense prefill-next prompt{idx}: first_id={_prefill_first_id} "
                    f"stop={_prefill_first_id in _stop_ids} "
                    f"top5={[int(x) for x in _prefill_topi.tolist()]}",
                    flush=True,
                )
                # Excludes the top-5 diagnostic, matching the DiffKV branch.
                prefill_time = forward_time

                peak_prefill_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                cur = prompt_len
                gen_ids = []
                # Pre-allocate static decode tensors
                _inp = torch.zeros((1, 1), dtype=torch.long, device=device)
                _pos = torch.zeros((1, 1), dtype=torch.long, device=device)
                t_decode_start = time.perf_counter()
                for _ in range(256):
                    nid_gpu = torch.argmax(last_logits_gpu)
                    nid = int(nid_gpu.item())  # one scalar D2H — unavoidable for stop check
                    if nid in _stop_ids:
                        break
                    gen_ids.append(nid)
                    _inp[0, 0] = nid
                    _pos[0, 0] = cur
                    out = model(_inp, position_ids=_pos, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                    last_logits_gpu = out.logits[0, -1].float()
                    cur += 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_time = time.perf_counter() - t_decode_start
                peak_decode_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

                generated_text = tok.decode(gen_ids)
                L   = model.config.num_hidden_layers
                Hkv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
                d   = model.config.hidden_size // model.config.num_attention_heads
                kv_vram = (L * prompt_len * Hkv * d * 2 * 2) / 1e9  # fp16 bytes × K+V

                results[f"prompt{idx}"] = {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "generated_tokens": len(gen_ids),
                    "prefill_time_s": prefill_time,
                    "prefill_forward_s": forward_time,
                    "prefill_compress_s": compress_time,
                    "prefill_chunk_size": CH,
                    "decode_time_s": decode_time,
                    "decode_tps": len(gen_ids) / decode_time if decode_time > 0 else 0.0,
                    "peak_prefill_vram_gb": peak_prefill_vram,
                    "prefill_forward_vram_gb": 0.0,
                    "prefill_compress_vram_gb": 0.0,
                    "peak_decode_vram_gb": peak_decode_vram,
                    "pool_physical_mb": 0.0,   # dense baseline has no DiffKV pool
                    "kv_cache_vram_gb": kv_vram,
                    "output_text": generated_text,
                }

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def generate_report(all_results, model_id):
    from tabulate import tabulate
    report_dir = os.path.join(REPO, "colab")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "nat_evaluation_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Neighborhood Attention Paper Evaluation Report ({model_id})\n\n")
        f.write("This report compares the performance, memory usage, and generation quality of standard **Dense Attention** against **Differential KV (DiffKV)** under various presets, early layer rank boosting, and factual store routing on an A100 GPU.\n\n")
        
        for p_idx in [1, 2]:
            p_key = f"prompt{p_idx}"
            f.write(f"## Prompt {p_idx} Performance & Resource Comparison\n\n")
            
            headers = ["Config", "Status", "Prefill Total (s)", "— Forward (s)", "— Compress (s)", "Decode TPS", "Peak Prefill VRAM (GB)", "Peak Decode VRAM (GB)", "KV Cache VRAM (GB)", "Gen Tokens"]
            rows = []

            for cfg_name, cfg_res in all_results.items():
                status_text = cfg_res.get("status", "success").upper()
                p_res = cfg_res.get(p_key, {})
                if status_text == "OOM":
                    rows.append([cfg_name, "OOM (Out of VRAM)", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                    continue
                rows.append([
                    cfg_name,
                    status_text,
                    f"{p_res.get('prefill_time_s', 0):.3f}s",
                    f"{p_res.get('prefill_forward_s', 0):.3f}s",
                    f"{p_res.get('prefill_compress_s', 0):.3f}s",
                    f"{p_res.get('decode_tps', 0):.2f}",
                    f"{p_res.get('peak_prefill_vram_gb', 0):.2f} GB",
                    f"{p_res.get('peak_decode_vram_gb', 0):.2f} GB",
                    f"{p_res.get('kv_cache_vram_gb', 0):.3f} GB",
                    str(p_res.get("generated_tokens", 0))
                ])
            
            f.write(tabulate(rows, headers=headers, tablefmt="github") + "\n\n")
            
        f.write("## Quality and Output Analysis\n\n")
        for p_idx in [1, 2]:
            p_key = f"prompt{p_idx}"
            f.write(f"### Responses to Prompt {p_idx}\n\n")
            
            for cfg_name, cfg_res in all_results.items():
                p_res = cfg_res.get(p_key, {})
                text = p_res.get("output_text", "").strip()
                word_count = len(text.split())
                
                f.write(f"#### Configuration: `{cfg_name}` ({word_count} words)\n")
                f.write(f"> {text}\n\n")
                
    print(f"Generated Markdown report at {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", help="run worker mode for a specific config")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="nat_eval_results.json")
    args = parser.parse_args()

    if args.worker:
        res = run_worker(args.worker, args.model)
        temp_file = f"temp_res_{args.worker}.json"
        with open(temp_file, "w") as f:
            json.dump(res, f)
        return

    configs = [
        "dense",
        "low_preset",
        "mid_preset",
        "high_preset",
        "early_boost",
        "factual_store",
        "combined"
    ]
    all_results = {}

    print(f"=== Starting Neighborhood Attention Paper Evaluation with {args.model} ===", flush=True)

    for cfg in configs:
        print(f"\n>>> Running configuration: {cfg}", flush=True)
        env = os.environ.copy()
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", cfg, "--model", args.model]
        
        p = subprocess.run(cmd, env=env)
        
        temp_file = f"temp_res_{cfg}.json"
        if os.path.exists(temp_file):
            with open(temp_file, "r") as f:
                res = json.load(f)
            os.remove(temp_file)
            all_results[cfg] = res
            
            for p_key in ["prompt1", "prompt2"]:
                p_res = res.get(p_key, {})
                print(
                    f"    {p_key} Success: tokens={p_res.get('generated_tokens')}, "
                    f"prefill={p_res.get('prefill_time_s',0):.2f}s "
                    f"(fwd={p_res.get('prefill_forward_s',0):.2f}s "
                    f"+ compress={p_res.get('prefill_compress_s',0):.2f}s, "
                    f"CH={p_res.get('prefill_chunk_size','?')}), "
                    f"tps={p_res.get('decode_tps',0):.1f}, "
                    # PEAK VRAM is the number that matters for "does DiffKV save RAM":
                    # torch.cuda.max_memory_allocated across prefill / decode, which
                    # includes weights + raw KV + pool + workspaces.  kv_mem is the
                    # analytic/logical store size only (misleadingly small — it does
                    # not count the pool's uniform-slot padding or transient buffers).
                    f"peak_prefill={p_res.get('peak_prefill_vram_gb',0):.2f}GB, "
                    f"after_fwd={p_res.get('prefill_forward_vram_gb',0):.2f}GB, "
                    f"after_comp={p_res.get('prefill_compress_vram_gb',0):.2f}GB, "
                    f"peak_decode={p_res.get('peak_decode_vram_gb',0):.2f}GB, "
                    f"pool={p_res.get('pool_physical_mb',0):.0f}MB, "
                    f"kv_logical={p_res.get('kv_cache_vram_gb',0):.3f}GB",
                    flush=True,
                )
        else:
            print(f"    Subprocess for {cfg} crashed or went Out-Of-Memory (OOM).", flush=True)
            all_results[cfg] = {
                "status": "OOM",
                "prompt1": {"output_text": "ERROR: CUDA OUT OF MEMORY (OOM) - Model footprint too large for GPU VRAM."},
                "prompt2": {"output_text": "ERROR: CUDA OUT OF MEMORY (OOM) - Model footprint too large for GPU VRAM."}
            }

    # Save raw results
    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote raw results to {out_path}", flush=True)

    # Generate visual markdown report
    generate_report(all_results, args.model)

if __name__ == "__main__":
    main()
