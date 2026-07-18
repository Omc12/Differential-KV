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

A researcher claims:

"Neighborhood Attention is simply Window Self-Attention with overlapping windows."

Evaluate this claim.

Identify why the claim may initially sound plausible, then explain precisely why it is incorrect.

Your explanation must distinguish:

- partitioning the feature map into windows,
- the attention span assigned to an individual token,
- boundary behavior,
- receptive-field expansion,
- and the role of shifted windows.

Do not use outside knowledge.
Do not discuss experimental accuracy, datasets, CUDA, or implementation speed.
Do not introduce mechanisms not explicitly stated in the supplied text.

Write one coherent technical explanation of 150 to 220 words."""

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
    """Footprint of the DiffKV cache, both logical (ideal) and physical (real).

    `store_used_bytes` is the LOGICAL number: what a perfectly packed store
    would hold.  It is what the paper quotes, and it is NOT what the GPU
    allocates.  `pool_physical_bytes` is the real pool allocation and is the
    only figure comparable to dense's KV bytes.

    Two bugs used to make the logical number meaningless on CUDA (both fixed in
    KVRuntimeManager.sessions, see that property's docstring): the block list
    was read from the wrong dict and the pool index was read under the wrong
    attribute name, so nb and res_tokens_used were always 0 and this returned
    exactly L * recency_window * kv_tok = 0.1007 GB for every preset.

    The block size here is the REAL block size (micro_block_size, 256 on CUDA),
    not mgr.block_size — that attribute is a hardcoded 64 (kv_runtime_manager
    line ~465) that the streaming path never uses, and it under-counted U while
    over-counting the block count's divisor.
    """
    L = mgr.num_layers
    Hkv = mgr.kv_heads
    d = mgr.head_dim
    fp16 = 2

    # Real per-block token capacity on the streaming path.
    B = int(getattr(mgr, "micro_block_size", 0) or getattr(mgr, "block_size", 64))
    # Rank actually stored per block.  The pool's rank dimension is the width
    # that write_block fills; _block_boost_rank can raise a block to
    # ceil(rank*1.5), which is why pool_rank is 1.5x the configured rank.
    pool = getattr(mgr, "native_pool", None)
    r = int(getattr(pool, "rank", None) or mgr.rank)

    kv_tok = Hkv * d * fp16 * 2
    lowrank_block = (B * r * 1                    # U (int8)
                     + 2 * Hkv * r * d * fp16     # V_K + V_V
                     + 2 * Hkv * d * fp16         # anchors K + V
                     + 8)

    s0 = mgr.sessions.get(sid)
    nb = s0["num_blocks"][0] if s0 else 0
    dl = s0["dense_lens"][0] if s0 else 0
    res_n0 = s0["comp_res_n"][0][:nb] if s0 else []
    res_tokens_used = int(sum(res_n0))

    store_used = L * (nb * lowrank_block + res_tokens_used * kv_tok + dl * kv_tok)

    # Physical: what the pool really allocated (uniform slots, lazy-grown).
    pool_physical = 0
    if pool is not None and hasattr(pool, "_pool_mb"):
        pool_physical = int(pool._pool_mb() * 1024 ** 2)

    dense_equiv = L * seq_len * kv_tok
    return {
        "store_used_bytes": store_used,
        "pool_physical_bytes": pool_physical,
        "dense_equiv_bytes": dense_equiv,
        "blocks_layer0": nb,
        "residual_tokens_layer0": res_tokens_used,
        "dense_window_tokens": dl,
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
    os.environ["DIFFKV_LAYER_ADAPTIVE_RANK"] = "0"
    os.environ["DIFFKV_STREAMING_COMPRESS"] = "0"

    # Default this eval to the A/B'd speed+memory combo via the single toggle
    # (the wrapper's _apply_fast_mode expands DIFFKV_FAST into the individual
    # flags; see its docstring).  setdefault so an explicit DIFFKV_FAST=0 or any
    # individual flag still wins.  DECODE_PRUNE is NOT bundled (confirmed dead
    # end).  The rank knobs it enables are fidelity-affecting — validate
    # test_niah.py before relying on FAST for number-heavy retrieval.
    os.environ.setdefault("DIFFKV_FAST", "1")

    is_compressed = (config_name != "dense")
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1" if is_compressed else "0"

    if is_compressed:
        if config_name == "low_preset":
            os.environ["DIFFKV_PRESET"] = "low"
        elif config_name == "adaptive_rank":
            os.environ["DIFFKV_PRESET"] = "low"
            os.environ["DIFFKV_LAYER_ADAPTIVE_RANK"] = "1"
        elif config_name == "adaptive_stream":
            os.environ["DIFFKV_PRESET"] = "low"
            os.environ["DIFFKV_LAYER_ADAPTIVE_RANK"] = "1"
            os.environ["DIFFKV_STREAMING_COMPRESS"] = "1"
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

        # NOTE (defaults audit): intentionally NOT calling apply_best_decode_defaults
        # here.  On CUDA it sets DIFFKV_SPARSE_BIAS=auto, but the fast fused
        # combined Triton decode kernel only runs when the bias is 0
        # (diffkv_attention.py: combined-path gate) — auto would silently drop
        # decode onto the slower separate path.  And DIFFKV_DECODE_CACHE=1 (the
        # "~2x tps" default) is a no-op on CUDA: CUDA's decode cache is the
        # separate, accuracy-gated DIFFKV_DECODE_CACHE_ENABLED (off by default).
        # So the "production decode defaults" would hurt, not help, CUDA here.
        # Keep the eval on the fast combined path (bias unset → 0.0).

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
    # Single prompt (the researcher-claim evaluation). One prompt per config keeps
    # the run short and the output easy to read/compare.
    all_prompts = [
        (idx, build_prompt(tok, paper_text, instr))
        for idx, instr in enumerate([PROMPT1_TEXT], 1)
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
                    #
                    # Fix 4 (opt-in): DIFFKV_STREAMING_COMPRESS=1 streams the
                    # compression per chunk (MLX-parity) to bound peak VRAM at
                    # long context — at the cost of later chunks attending the
                    # lossy form of far-back blocks.  A/B against the default.
                    if os.environ.get("DIFFKV_STREAMING_COMPRESS", "0") == "1" \
                            and hasattr(mgr, "compress_deferred_prefill_blocks"):
                        mgr.compress_deferred_prefill_blocks(sid)
                        # MLX-parity: after compressing out-of-window blocks, the
                        # raw active_k/v they held is dereferenced.  Return that
                        # freed memory to the allocator now (MLX calls
                        # mx.clear_cache() here) so peak VRAM reflects the bounded
                        # recency-window + pool footprint instead of the full
                        # prompt's raw KV — the whole point of streaming at 64k+.
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

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

                # ── Rank-boost rate ──────────────────────────────────────
                # _block_boost_rank raises a block to ceil(rank*1.5) when its
                # text has any digit / math char / definition phrase.  On
                # technical prose that predicate fires on ~100% of blocks (a
                # single hyphen in "self-attention" matches), so the "boost" is
                # really a flat 1.5x on rank — 1.5x pool bytes and 1.5x rSVD
                # work vs MLX, which has no SVD-rank boost at all.  Print the
                # rate so this is visible instead of inferred.
                _bs = getattr(mgr, "_rank_boost_stats", {}).get(sid)
                if _bs and _bs.get("total"):
                    _pct = 100.0 * _bs["boosted"] / _bs["total"]
                    print(f"[DIAG] rank boost fired on {_bs['boosted']}/{_bs['total']} "
                          f"blocks = {_pct:.1f}%  (100% => flat 1.5x rank; "
                          f"set DIFFKV_RANK_BOOST=off for MLX parity)", flush=True)

                # Verbose per-block state dump removed — the kv_logical / block-
                # accounting investigation it served is resolved.  Set
                # DIFFKV_DIAG=1 for the runtime's own [DIAG compress_deferred] trace
                # if block eligibility ever needs re-checking.

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
                    # The number that is actually comparable to dense's KV bytes.
                    # kv_cache_vram_gb is a logical ideal; this is what the GPU holds.
                    "kv_physical_gb": kv.get("pool_physical_bytes", 0) / 1e9,
                    "kv_dense_equiv_gb": kv.get("dense_equiv_bytes", 0) / 1e9,
                    "kv_blocks_layer0": kv.get("blocks_layer0", 0),
                    "kv_residual_tokens_layer0": kv.get("residual_tokens_layer0", 0),
                    "output_text": generated_text,
                }

                # Release THIS prompt's session before the next one runs.
                # The loop previously only ever cleared the fresh, not-yet-created
                # sid ("prompt_2" on iteration 2 is a no-op), so prompt_1's pool
                # slots + block/SRL state were never freed.  prompt_2 then
                # allocated its own on top, doubling `pool` (1039MB -> 2065MB) and
                # inflating peak VRAM.  clear_session() returns the pool slots to
                # the free list (native_pool.free_block) so the next prompt reuses
                # them instead of growing the pool.  Dense's per-prompt state is
                # already independent (fresh past_key_values), so this makes the
                # DiffKV rows a like-for-like single-session comparison.
                mgr.clear_session(sid)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        
        for p_idx in [1]:
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
        for p_idx in [1]:
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
        "adaptive_rank",
        "adaptive_stream",
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
            
            for p_key in ["prompt1"]:
                p_res = res.get(p_key, {})

                # PEAK VRAM is the number that matters for "does DiffKV save RAM":
                # torch.cuda.max_memory_allocated across prefill / decode, which
                # includes weights + raw KV + pool + workspaces.
                #
                # kv_logical is the analytic/ideal store size.  kv_phys is the
                # pool's real allocation and is the ONLY figure comparable to the
                # dense KV bytes it replaces — quoting kv_logical as the
                # compression ratio overstates it by ~10x, because the pool pays
                # for uniform worst-case slots on every block.
                _phys = p_res.get("kv_physical_gb", 0.0)
                _dense = p_res.get("kv_dense_equiv_gb", 0.0)
                if _phys > 0:
                    kv_str = (
                        f"kv_logical={p_res.get('kv_cache_vram_gb',0):.3f}GB"
                        f"[blk0={p_res.get('kv_blocks_layer0',0)}], "
                        f"kv_phys={_phys:.3f}GB vs dense {_dense:.3f}GB "
                        f"= {_dense / _phys:.2f}x REAL"
                    )
                else:
                    kv_str = f"kv_logical={p_res.get('kv_cache_vram_gb',0):.3f}GB"

                print(
                    f"    {p_key} Success: tokens={p_res.get('generated_tokens')}, "
                    f"prefill={p_res.get('prefill_time_s',0):.2f}s "
                    f"(fwd={p_res.get('prefill_forward_s',0):.2f}s "
                    f"+ compress={p_res.get('prefill_compress_s',0):.2f}s, "
                    f"CH={p_res.get('prefill_chunk_size','?')}), "
                    f"tps={p_res.get('decode_tps',0):.1f}, "
                    f"peak_prefill={p_res.get('peak_prefill_vram_gb',0):.2f}GB, "
                    f"after_fwd={p_res.get('prefill_forward_vram_gb',0):.2f}GB, "
                    f"after_comp={p_res.get('prefill_compress_vram_gb',0):.2f}GB, "
                    f"peak_decode={p_res.get('peak_decode_vram_gb',0):.2f}GB, "
                    f"pool={p_res.get('pool_physical_mb',0):.0f}MB, "
                    + kv_str,
                    flush=True,
                )
                # Show the actual generated text so runs are judged on OUTPUT, not
                # just metrics — essential for A/B'ing the numerics-changing paths
                # (Gram, contiguous/un-rotate prefill, and any future CUDA-graph
                # decode).  Full text unless DIFFKV_EVAL_OUTPUT_CHARS caps it.
                _otext = p_res.get("output_text", "").strip().replace("\n", " ")
                try:
                    _cap = int(os.environ.get("DIFFKV_EVAL_OUTPUT_CHARS", "0"))
                except ValueError:
                    _cap = 0
                if _cap > 0 and len(_otext) > _cap:
                    _otext = _otext[:_cap] + " …"
                print(f"      ↳ output: {_otext}", flush=True)
        else:
            print(f"    Subprocess for {cfg} crashed or went Out-Of-Memory (OOM).", flush=True)
            all_results[cfg] = {
                "status": "OOM",
                "prompt1": {"output_text": "ERROR: CUDA OUT OF MEMORY (OOM) - Model footprint too large for GPU VRAM."},
                "prompt2": {"output_text": "ERROR: CUDA OUT OF MEMORY (OOM) - Model footprint too large for GPU VRAM."}
            }

    # ── Where does the time and storage go? ──────────────────────────────────
    # One table per run so dense vs each preset is comparable at a glance:
    #   TIME  — prefill forward, compression, and decode (tps).
    #   VRAM  — peak prefill (the spike), peak decode, and the KV store
    #           (pool physical vs the dense KV it replaces).
    print("\n" + "=" * 100, flush=True)
    print("TIME / STORAGE BREAKDOWN (prompt1)", flush=True)
    print("=" * 100, flush=True)
    _hdr = (f"{'config':<14} {'fwd_s':>7} {'comp_s':>7} {'dec_s':>7} {'tps':>6} "
            f"{'peak_pf':>8} {'peak_dec':>8} {'pool_MB':>8} {'kv_phys':>8} {'vs_dense':>9}")
    print(_hdr, flush=True)
    print("-" * 100, flush=True)
    _dense_kv = None
    for cfg_name, cfg_res in all_results.items():
        p = cfg_res.get("prompt1", {})
        if p.get("status") != "success":
            print(f"{cfg_name:<14} {'OOM/ERROR':>7}", flush=True)
            continue
        _kv_phys = p.get("kv_physical_gb", 0.0)
        _dense_eq = p.get("kv_dense_equiv_gb", 0.0)
        if cfg_name == "dense":
            _dense_kv = _dense_eq or 0.0
        _store = _kv_phys if _kv_phys > 0 else p.get("kv_cache_vram_gb", 0.0)
        _ratio = (_dense_kv / _store) if (_store > 0 and _dense_kv) else 0.0
        print(
            f"{cfg_name:<14} "
            f"{p.get('prefill_forward_s',0):>7.2f} "
            f"{p.get('prefill_compress_s',0):>7.2f} "
            f"{p.get('decode_time_s',0):>7.2f} "
            f"{p.get('decode_tps',0):>6.1f} "
            f"{p.get('peak_prefill_vram_gb',0):>7.2f}G "
            f"{p.get('peak_decode_vram_gb',0):>7.2f}G "
            f"{p.get('pool_physical_mb',0):>8.0f} "
            f"{_store:>7.2f}G "
            f"{(str(round(_ratio,2))+'x') if _ratio else '—':>9}",
            flush=True,
        )
    print("=" * 100, flush=True)

    # Save raw results
    out_path = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote raw results to {out_path}", flush=True)

    # Generate visual markdown report
    generate_report(all_results, args.model)

if __name__ == "__main__":
    main()
