import os
import sys
if os.environ.get("DKV_FORCE_PYTORCH") == "1" and sys.platform != "darwin":
    sys.modules["dkv_core"] = None
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
# Bound the CUDA caching allocator's fragmentation. Prefill compresses every
# block on every layer, allocating and freeing thousands of short-lived tensors
# of many different sizes, and the default segment allocator cannot reuse a freed
# segment for a differently-sized request. Measured on Qwen2.5-1.5B at 32k: 5.44
# GB of live tensors against 16.2 GB RESERVED, which on a 12 GB card spills into
# WDDM shared host memory -- that spill is why prefill was slow and why DKV took
# ~2x dense's real VRAM despite holding a smaller KV.
#
# expandable_segments lets one segment grow and shrink instead, so mixed sizes
# share it: reserved 16.2 -> 8.9 GB, real device use 11.0 -> 8.5 GB, and TTFT
# 21.2 -> 15.7 s at 32k.
#
# Must be set before the caching allocator initialises (first CUDA allocation);
# importing torch alone does not initialise it, so setting it here is in time.
# setdefault so an explicit caller value still wins. Note this is the one
# allocator mode that has historically interacted badly with CUDA graph capture,
# which is why the graph path stays opt-in.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
"""
runtime/hf_dkv_wrapper.py

HuggingFace model wrapper for Differential KV.
Integrates KVRuntimeManager with AutoModelForCausalLM.

Mac/MPS: device is auto-detected (CUDA → MPS → CPU).
  - torch.compile uses 'aot_eager' backend on MPS (no CUDAGraph dependency).
  - torchao quantization is skipped on MPS where not yet supported.
  - All CUDA-specific calls are routed through mac_utils helpers.
"""

import torch
import torch.nn as nn
import re
import sys
from collections import Counter
from typing import Optional, List, Tuple, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from native_core.kv_runtime_manager import KVRuntimeManager

from native_core.sparse_decode.triton_fused_decode import TritonDKV
from runtime.dkv_attention import apply_dkv_attention_patch
from runtime.dkv_backend import register_dkv_backend, bind_kv_manager
from serving.query_span import extract_query_token_ids as _extract_query_token_ids
from serving.query_span import pinned_blocks_from_prompt as _pinned_blocks_from_prompt


try:
    from native_core.graph_runtime.static_decode_graph import CUDAGraphDecodeRunner
    _HAS_CUDA_GRAPH_RUNNER = True
except ImportError:
    _HAS_CUDA_GRAPH_RUNNER = False

try:
    from native_core.mac_utils import (
        get_best_device as _get_best_device,
        get_compile_backend as _get_compile_backend,
        get_compile_mode as _get_compile_mode,
        has_cuda as _has_cuda,
        has_mps as _has_mps,
        is_apple_silicon as _is_apple_silicon,
    )
except ImportError:
    def _get_best_device(): return "cuda" if torch.cuda.is_available() else "cpu"
    def _get_compile_backend(): return "inductor" if torch.cuda.is_available() else "aot_eager"
    def _get_compile_mode(): return "reduce-overhead" if torch.cuda.is_available() else "default"
    def _has_cuda(): return torch.cuda.is_available()
    def _has_mps(): return getattr(getattr(torch, 'backends', None), 'mps', None) and torch.backends.mps.is_available()
    def _is_apple_silicon(): return False

import traceback

def _patch_tensor_sync_barriers():
    orig_item = torch.Tensor.item
    orig_cpu = torch.Tensor.cpu
    orig_tolist = torch.Tensor.tolist

    _in_sync_check = False

    def patched_item(self):
        nonlocal _in_sync_check
        if not _in_sync_check and self.device.type != "cpu":
            _in_sync_check = True
            print("[DKV_SYNC_DEBUG] WARNING: Synchronization barrier triggered by .item() call!")
            traceback.print_stack(limit=5)
            _in_sync_check = False
        return orig_item(self)

    def patched_cpu(self, *args, **kwargs):
        nonlocal _in_sync_check
        if not _in_sync_check and self.device.type != "cpu":
            _in_sync_check = True
            print("[DKV_SYNC_DEBUG] WARNING: Synchronization barrier triggered by .cpu() call!")
            traceback.print_stack(limit=5)
            _in_sync_check = False
        return orig_cpu(self, *args, **kwargs)

    def patched_tolist(self):
        nonlocal _in_sync_check
        if not _in_sync_check and self.device.type != "cpu":
            _in_sync_check = True
            print("[DKV_SYNC_DEBUG] WARNING: Synchronization barrier triggered by .tolist() call!")
            traceback.print_stack(limit=5)
            _in_sync_check = False
        return orig_tolist(self)

    # Patch them
    torch.Tensor.item = patched_item
    torch.Tensor.cpu = patched_cpu
    torch.Tensor.tolist = patched_tolist
    print("[DKV] DKV_SYNC_DEBUG sync barrier checks enabled.")


# ── Memory Reduction Helpers ──────────────────────────────────────────────────

def _get_rss_mb() -> float:
    """Return current process RSS in megabytes."""
    try:
        import psutil, os as _os
        return psutil.Process(_os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        return 0.0


def _trim_python_heap() -> None:
    """
    Force Python's malloc arena to return unused virtual pages to the OS.

    HuggingFace model loading creates ~2–3 GB of temporary CPU tensors
    (FP32 weight copies, conversion intermediates) that are freed but whose
    virtual address space stays in Python's internal free-list. This call
    reclaims that space so macOS stops counting it as resident/swapped memory.

    Expected result: Python heap virtual 4.8 GB → ~1.2 GB after a 0.5B load.
    """
    import gc, ctypes, sys

    rss_before = _get_rss_mb()

    # Triple GC pass to break cyclic references left by HF model loading
    gc.collect(); gc.collect(); gc.collect()

    # Ask the OS to reclaim free malloc arena pages
    if sys.platform == "linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
    elif sys.platform == "darwin":
        # macOS does not export malloc_trim; the correct call is malloc_zone_pressure_relief
        # (available since macOS 10.11). It compacts all default malloc zones and returns
        # unused virtual pages to the OS — identical semantics to malloc_trim(0) on Linux.
        try:
            libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            # malloc_zone_pressure_relief(zone=NULL, goal=0) → relieves all zones
            libsystem.malloc_zone_pressure_relief(None, ctypes.c_size_t(0))
        except (OSError, AttributeError):
            # Symbol not found on older macOS or sandbox restriction — not fatal
            pass

    # Python 3.13+ explicit free-list compaction (available in 3.14)
    if hasattr(sys, "_compact_freelists"):
        sys._compact_freelists()

    rss_after = _get_rss_mb()
    saved = rss_before - rss_after
    print(f"[DKV] Heap trimmed. RSS: {rss_before:.0f} MB → {rss_after:.0f} MB "
          f"(saved {max(0.0, saved):.0f} MB)")


def _clear_cpu_grad_state(model: "torch.nn.Module") -> None:
    """
    Disable gradient tracking and free any lingering .grad tensors.

    HuggingFace models load with requires_grad=True by default. For inference
    this is dead weight — the autograd graph consumes memory with no benefit.
    Disabling it also prevents accidental gradient accumulation.
    """
    import gc
    torch.autograd.set_grad_enabled(False)
    for param in model.parameters():
        param.requires_grad_(False)
        if hasattr(param, "grad") and param.grad is not None:
            param.grad = None
    gc.collect()


def _clear_cpu_param_copies(model: "torch.nn.Module", device: str) -> None:
    """
    After model.to(device), audit for stray CPU parameters and flush device cache.

    In some HuggingFace versions, the CPU weight storage is retained even after
    the tensor has been moved to MPS/CUDA. Flushing the cache reclaims ~1 GB.
    """
    import gc
    stray_count = 0
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            stray_count += 1
    if stray_count:
        print(f"[DKV] WARNING: {stray_count} parameters still on CPU after to({device}) — "
              "possible HF version issue.")

    gc.collect()
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _configure_mps_memory(memory_fraction: Optional[float] = None) -> None:
    """
    Cap MPS allocator and start a lightweight daemon that releases the cache
    under memory pressure.

    memory_fraction: fraction of system unified memory MPS may use before
    triggering GC. Set via DKV_MPS_MEMORY_FRACTION env var or config dict.
    If None, no artificial cap is set (recommended to avoid allocator OOMs).

    The daemon thread polls RSS every 5 seconds. When RSS > 3 GB it calls
    torch.mps.empty_cache() + gc.collect(). This prevents the OS from
    swapping GPU memory to disk during long conversations.
    """
    import gc, os, threading

    if not (hasattr(torch, "backends") and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return

    # Cap MPS reservation only if explicitly requested
    if memory_fraction is not None:
        try:
            torch.mps.set_per_process_memory_fraction(memory_fraction)
            print(f"[DKV] MPS memory fraction capped at {memory_fraction:.0%} of system RAM.")
        except Exception as e:
            print(f"[DKV] WARNING: Could not set MPS memory fraction: {e}")
    else:
        print("[DKV] MPS memory fraction: unlimited (no artificial cap applied).")

    # RSS-based pressure relief daemon
    rss_threshold_mb = float(os.environ.get("DKV_MPS_RSS_THRESHOLD_MB", "3000"))

    def _pressure_monitor():
        import time
        while True:
            time.sleep(5.0)
            try:
                rss = _get_rss_mb()
                if rss > rss_threshold_mb:
                    gc.collect()
            except Exception:
                pass

    t = threading.Thread(target=_pressure_monitor, daemon=True, name="dkv-mps-pressure")
    t.start()
    print(f"[DKV] MPS pressure monitor started (threshold: {rss_threshold_mb:.0f} MB RSS).")


def _configure_cuda_allocator() -> None:
    """
    Set conservative CUDA allocator options to reduce fragmentation.

    garbage_collection_threshold:0.6 — trigger GC when 60% of reserved memory
      is actively allocated (vs default 80%), reducing peak fragmentation.
    max_split_size_mb:128 — largest block the caching allocator will split.
      Smaller splits mean fewer huge stranded blocks, lower peak VRAM.

    This is a setdefault: a caller that already exported PYTORCH_CUDA_ALLOC_CONF
    (run_nat_eval.py sets expandable_segments:True at import) keeps its value.
    Report what is actually in effect rather than the defaults we asked for —
    the old unconditional message claimed gc_threshold/max_split_size were
    configured even when the caller's setting had already won.

    TF32: `set_float32_matmul_precision('high')` lets every fp32 matmul use the
    Ampere+ TF32 tensor cores.  This does NOT help compress (cuSOLVER-bound) and
    does NOT touch the fp16 prefill attention, but it DOES speed up the fp32
    DECODE reconstruction JIT kernels (`_reconstruct_and_score` etc.) — the ones
    PyTorch's own "TensorFloat32 ... not enabled" warning fires on.  That is a
    decode-tps win.  It slightly perturbs fp32 accumulation (TF32 has a 10-bit
    mantissa), so generated tokens can shift a little; that is the speed/accuracy
    trade this project has opted into.  Opt out with DKV_TF32=0.  The narrow
    compress-only `_tf32_matmul` (lowrank.py) remains as belt-and-suspenders.
    """
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "garbage_collection_threshold:0.6,max_split_size_mb:128"
    )
    print(f"[DKV] CUDA allocator config: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    if os.environ.get("DKV_TF32", "1") != "0":
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.set_float32_matmul_precision("high")
                _torch.backends.cuda.matmul.allow_tf32 = True
                _torch.backends.cudnn.allow_tf32 = True
                print("[DKV] TF32 enabled globally (float32_matmul_precision=high) "
                      "— speeds fp32 decode reconstruction kernels. DKV_TF32=0 to disable.")
        except Exception as _e:
            print(f"[DKV] Could not enable global TF32: {_e}")

    _apply_fast_mode()


def _apply_fast_mode() -> None:
    """DKV_FAST=1 — one toggle for the A/B'd speed+memory CUDA combo.

    Bundles (via setdefault, so an explicit flag still wins):
      DKV_COMPRESS_GRAM_SVD=1   recon-equivalent SVD (safe)
      DKV_CONTIGUOUS_PREFILL=1  forward faster than dense (recon-equivalent)
      DKV_CONTIG_UNROTATE=1     1x-memory prefill, peak ~dense (fp16-equivalent)
      DKV_RANK_BOOST=off        the boost fired on ~100% of blocks (a fake flat
                                   1.5x); off = the configured rank + pool_rank
                                   48->32 (VRAM ratio up to ~3x)
      DKV_RSVD_MAX_RPROJ=32     enables the batched-compress cuSOLVER cliff
                                   (compress 6s->2.6s)

    ⚠ The last two are FIDELITY-AFFECTING (rank cap + 0 oversamples).  The
    content-aware rank boost existed to give DIGIT/number blocks extra rank, so
    validate needle recall before trusting FAST on number-heavy retrieval:
        DKV_MODEL=<model> python -m pytest ACTIVE_RUNTIME/tests/test_niah.py -v
    run WITH and WITHOUT DKV_FAST and confirm the 6-digit needle still returns.
    """
    if os.environ.get("DKV_FAST", "0") != "1":
        return
    for _k, _v in (
        ("DKV_COMPRESS_GRAM_SVD", "1"),
        ("DKV_CONTIGUOUS_PREFILL", "1"),
        ("DKV_CONTIG_UNROTATE", "1"),
        ("DKV_RANK_BOOST", "off"),
        ("DKV_RSVD_MAX_RPROJ", "32"),
    ):
        os.environ.setdefault(_k, _v)
    print("[DKV] DKV_FAST=1 — batched-compress cliff + contiguous 1x prefill "
          "(recon-equivalent SVD; rank cap 32 is fidelity-affecting — validate "
          "test_niah.py). DECODE_PRUNE deliberately NOT bundled (dead end).")


def _sample_logits(logits, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0.01:
        return torch.argmax(logits, dim=-1)
    else:
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        if top_p < 1.0:
            s_probs, s_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(s_probs, dim=-1)
            mask = (cum - s_probs) > top_p
            s_probs[mask] = 0.0
            s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True)
            sample = torch.multinomial(s_probs, 1)
            return s_idx.gather(-1, sample).squeeze(-1)
        else:
            return torch.multinomial(probs, 1).squeeze(-1)

if torch.cuda.is_available():
    try:
        _compiled_sample_fn = torch.compile(
            _sample_logits,
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False,
        )
    except Exception:
        _compiled_sample_fn = _sample_logits
else:
    _compiled_sample_fn = _sample_logits

def _normalize_references(text: str) -> str:
    """Normalise citation-list formatting inconsistencies produced by the model."""
    lines = text.split('\n')
    
    # 1. Search for a reference header line
    header_re = re.compile(r'\b(references?|bibliography|works\s+cited|reference\s+list|sources|citations)\b', re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if len(line) <= 100 and header_re.search(line):
            header_idx = i
    
    # 2. Find matching reference entries
    ref_entry_re = re.compile(r'^(?:[iI]n\s+)?(?:[*\-•]\s*)?\[\d+\]')
    unambiguous_re = re.compile(r'^(?:[*\-•]\s*)?\[\d+\]')
    
    matching_indices = []
    unambiguous_indices = []
    for i, line in enumerate(lines):
        if header_idx is not None and i <= header_idx:
            continue
        stripped = line.strip()
        if ref_entry_re.match(stripped):
            matching_indices.append(i)
            if unambiguous_re.match(stripped):
                unambiguous_indices.append(i)
                
    if header_idx is not None and not matching_indices:
        matching_indices = []
        unambiguous_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ref_entry_re.match(stripped):
                matching_indices.append(i)
                if unambiguous_re.match(stripped):
                    unambiguous_indices.append(i)
        header_idx = None

    if not matching_indices:
        return text
        
    if header_idx is not None:
        ref_start_idx = header_idx + 1
    elif unambiguous_indices:
        ref_start_idx = unambiguous_indices[0]
    else:
        return text
                
    body = '\n'.join(lines[:ref_start_idx])
    ref_block = '\n'.join(lines[ref_start_idx:])
    
    pattern = re.compile(
        r'^\s*'
        r'(?:[iI]n\s+)?'
        r'(?:[*\-•]\s*)?'
        r'(\[\d+\])'
        r'(?:,\s*|\.\s*|\s+)?',
        re.MULTILINE
    )
    normalized_ref_block = pattern.sub(r'\1 ', ref_block)
    
    if body:
        return body + '\n' + normalized_ref_block
    return normalized_ref_block

import runtime.dkv_attention as _dkv_attn_mod  # noqa: E402
from native_core.sparse_decode.remat_cache import (  # noqa: E402
    remat_interval as _dkv_remat_interval,
)


class PyTorchDKVHFWrapper:
    """
    Wraps a HuggingFace model to use Differential KV cache.
    """
    def __init__(
        self, 
        model_id: str,
        config: Dict[str, Any],
        device: str = None,   # None → auto-detect (CUDA / MPS / CPU)
        quantization_config: Any = None,
        torch_dtype: torch.dtype = None,  # None → auto (fp16 on GPU/MPS, bf16 on CPU)
        lazy: bool = False,
    ):
        self.model_id = model_id
        self.config = config or {}
        self._quantization_config = quantization_config
        self._torch_dtype_arg = torch_dtype
        self.device = device if device is not None else _get_best_device()
        self.lazy = lazy
        
        self.mode = self.config.get("mode", "fp16")
        self.block_size = self.config.get("block_size", 256)      # S=256 → 5.2× compression
        # Rank 32 matches the MLX wrapper (mlx_dkv_wrapper.py:4493) and the
        # paper's config of record.  This path defaulted to 16, which is the
        # exact value already diagnosed as a ~43% needle-recall floor in the
        # native runtime — CUDA was the last runtime still shipping it.
        # Take the rank from the PRESET when the caller did not pin one. This
        # wrapper fixes self.rank before KVRuntimeManager builds its DKVConfig,
        # so a hardcoded default here silently overrode the preset: DKV_PRESET=high
        # resolved svd_energy=0.99999 but still ran rank 32, and the ceiling caps
        # the energy target -- the quality preset could not actually reach the
        # fidelity it asks for. Explicit config["rank"] still wins.
        if "rank" in self.config:
            self.rank = self.config["rank"]
        else:
            try:
                from native_core.config import DKVConfig as _DKVCfg
                self.rank = int(getattr(_DKVCfg(self.config), "rank", 32))
            except Exception:                                      # noqa: BLE001
                self.rank = 32
        # 1024, not 256. Blocks of 256 lose distractor-heavy retrieval: on
        # colab/linkbench_cuda.py (16 near-identical "The X Institute is located
        # in Y" sentences, answer graded on attribution, 24 seeds at 16k on
        # Qwen3.5-2B) a dense control scores 24/24 and DKV scored 14/24. Block
        # size is the only thing that moved it:
        #
        #     128 -> 11/24    256 -> 14/24    512 -> 15/24    1024 -> 24/24
        #
        # Nothing else did: routing K (0/32/default), residual budget
        # (32/128/224), recency window (512/4096) and SVD rank (32/96) all left
        # it at EXACTLY 14/24. Needle recall is unaffected (9/9 + 9/9
        # determinism at 1024, same as at 256), and the needle benchmark cannot
        # see the gap at all because one unique code in bland filler has no
        # confusable distractors.
        #
        # Cost: peak tensor bytes +0.10 GB and TTFT unchanged at 32k. The decode
        # effect is not resolvable here -- interleaved repeats of the SAME config
        # spanned 14.85-21.56 tok/s -- so it is somewhere between nil and ~16%,
        # and colab/bench_decode_paired.py cannot settle it because block size is
        # fixed when the manager is constructed rather than read per call.
        # 1024. Chosen for LINKAGE, which is the property DKV exists to have.
        #
        # Full sweep, Qwen3.5-2B accuracy at 16k, Qwen2.5-1.5B prefill/VRAM at
        # 32k. linkbench = colab/linkbench_cuda.py, 24 seeds, one fact among 16
        # near-identical distractors, graded on attribution:
        #
        #   block  linkbench  synthesis   TTFT     peak_alloc  needles
        #   256    14/24      46.7 (8/2)  15.17 s  5.44 GB     9/9
        #   512    15/24      50.0 (6/3)  11.58 s  5.10 GB     9/9
        #   1024   24/24      30.0 (6/1)  11.43 s  4.96 GB     9/9
        #   1536   -          26.7 (5/1)  -        -           -
        #   2048   24/24*     33.3 (4/2)  -        -           -
        #   dense  24/24      60.0 (9/3)   5.70 s  -           -
        #                                          (* measured at 32k)
        #
        # 1024 is the smallest block that reaches DENSE PARITY on distractor
        # retrieval, and it is also the best point measured for prefill and VRAM.
        # It costs synthesis, 50.0 -> 30.0.
        #
        # That is a real trade and it is taken deliberately: retrieving the right
        # fact from a document full of similar ones is the workload DKV is for,
        # and at 512 it lost 9 of 24 such lookups that dense got right. Set
        # micro_block_size=512 for synthesis-shaped work.
        #
        # The trade EXISTS because nothing else bridges the two metrics. Measured
        # and rejected: rank scaled with block size (1024 at rank 32/64/128 all
        # give synthesis 30.0, so the loss is not per-token fidelity), routing
        # coverage (DKV_TOPK_FRAC 0.0/0.5/1.0 and DKV_TOPK_BLOCKS=0 leave
        # linkbench unmoved), residual budget (32/128/224) and recency window
        # (512/4096). Distractor retrieval tracks the NUMBER of blocks the
        # context is split into -- ~15 blocks scores 24/24 however it got there
        # (256@4k, 1024@16k, 2048@32k), ~58 blocks scores 14/24 -- so what a
        # larger block buys is an association staying inside ONE unit.
        #
        # The real fix is dual-scale storage: the same content compressed at both
        # granularities with attention seeing both. Because routing is provably
        # irrelevant here, a multi-scale router would not help; both scales have
        # to reach attention. See ACTIVE_RUNTIME/docs/cuda_port_record.md for the shape of it.
        self.micro_block_size = self.config.get("micro_block_size", 1024)
        
        self.local_files_only = (
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            or self.config.get("local_files_only", False)
        )
        if self.local_files_only:
            print("[DKV] Offline mode active: loading model/tokenizer from local cache only.")

        print(f"[DKV] Lazy-initializing tokenizer for model {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            local_files_only=self.local_files_only
        )
        self._alphanumeric_tokens = {}
        
        # Initialize stop token IDs from tokenizer
        self.stop_token_ids = set()
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            self.stop_token_ids.update(eos_id)
        elif isinstance(eos_id, int):
            self.stop_token_ids.add(eos_id)
            
        special_words = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
        for word in special_words:
            tok_id = self.tokenizer.convert_tokens_to_ids(word)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tok_id)

        self.model = None
        self.manager = None
        self.active_session = None

        if not self.lazy:
            self.ensure_loaded()

    def ensure_loaded(self):
        if self.model is not None:
            return

        model_id = self.model_id
        config = self.config
        device = self.device
        quantization_config = self._quantization_config
        
        torch_dtype = self._torch_dtype_arg
        if torch_dtype is None:
            if self.device in ("cuda", "mps"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.bfloat16

        print(f"[DKV] Loading model weights on demand: {model_id} (device={self.device}, dtype={torch_dtype})...")
        
        # ── Preset-aware Auto-Quantization ──
        preset = config.get("preset", os.environ.get("DKV_PRESET", "mid")).lower()

        # ── MLX parity: quality presets opt into Context-Aware Decoding (CAD) ──
        # Mirrors mlx_dkv_wrapper (high/quality/max → DKV_CAD_ALPHA=0.5,
        # DKV_CAD_MAX_STEPS=32).  CAD is already implemented in this wrapper
        # (the PyTorch/CUDA port in generate()); it was just never auto-enabled
        # per preset like MLX.  It contrasts each step's full-context logits
        # against a prior-only stream to pull the decoder off its pretrained
        # prior onto the document's relation (relational-edge fidelity), capped
        # to DKV_CAD_MAX_STEPS tokens so it amortizes to ~0 on long
        # generations.  Explicit env always wins (setdefault).
        if preset in ("high", "quality", "max"):
            os.environ.setdefault("DKV_CAD_ALPHA", "0.5")
            os.environ.setdefault("DKV_CAD_MAX_STEPS", "32")
            print(f"[DKV] {preset} preset: Context-Aware Decoding on "
                  f"(alpha={os.environ['DKV_CAD_ALPHA']}, "
                  f"max_steps={os.environ['DKV_CAD_MAX_STEPS']}) — MLX parity")

        if preset == "low" and not config.get("quantization") and not os.environ.get("DKV_QUANTIZATION"):
            if self.device == "cuda":
                config["quantization"] = "nf4"
                print("[DKV] Low preset + CUDA: auto-enabling 4-bit NF4 quantization (bitsandbytes) to save VRAM")
            elif self.device == "mps":
                print("[DKV] Low preset + MPS: running in FP16 to avoid torchao NaN/stability issues on MPS")

        # ── 4-bit NF4 loading (BitsAndBytes) ──────────────────────────────────
        _quant_type_early = config.get("quantization") or os.environ.get("DKV_QUANTIZATION", "")
        if (quantization_config is None
                and _quant_type_early == "nf4"
                and _has_cuda()):
            try:
                from transformers import BitsAndBytesConfig as _BnBConfig
                quantization_config = _BnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                torch_dtype = torch.bfloat16
                print("[DKV] 4-bit NF4 quantization enabled (BitsAndBytes).")
            except ImportError:
                print("[DKV] WARNING: bitsandbytes not installed — falling back to fp16.")

        if _has_cuda():
            _configure_cuda_allocator()

        # Clean Code Integration: AttentionInterface path (Mode A default)
        attn_impl_kwarg = {}
        # DEFAULT "0" -- the MONKEYPATCH path (Path A).
        #
        # Path A runs the FUSED decode kernel (Triton on CUDA, the Metal shader
        # on MPS), which is what MLX does (DKV_DECODE_FUSED=1 by default there).
        # The AttentionInterface backend (dkv_backend.py, Path B) does NOT: it
        # has no reference to fused_decode/triton anywhere and falls through to a
        # plain F.scaled_dot_product_attention, so it pays DKV's Python-side
        # block bookkeeping and gets none of its kernel.
        #
        # Measured on an RTX PRO 4000, Qwen3.5-2B, 13k ctx, preset mid:
        #   Path B -> 4.3 tps (231.7 ms/token), of which only 33.4 ms/token is
        #   GPU work -- 86%% of wall time is CPU dispatch -- and the profiler's
        #   "dkv" bucket is 0.0 ms, i.e. the DKV kernel contributed nothing.
        #
        # This default used to be "1" here and in dkv_attention.py while
        # dkv_backend.py's own header documented "0", so the effective default
        # was the path without the kernel. Set =1 to opt back into Path B.
        if os.environ.get("DKV_USE_ATTENTION_INTERFACE", "0") == "1" or config.get("use_attention_interface", False):
            register_dkv_backend()
            attn_impl_kwarg["attn_implementation"] = "dkv"

        if device == "mps" and quantization_config is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=self.local_files_only,
                **attn_impl_kwarg,
            ).to(device)
        elif device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="mps",
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=self.local_files_only,
                **attn_impl_kwarg,
            )
        elif quantization_config is None and str(device).startswith("cuda"):
            # device_map, NOT load-then-.to(device).
            #
            # The .to(device) form materialises every weight on the CPU first and
            # the host pages stay resident afterwards: measured on Qwen3.5-2B,
            # RSS went 0.82 -> 4.97 GB across this one call and stayed there,
            # against 2.07 GB total for the same model loaded with device_map.
            # That is ~4 GB of system memory held for nothing, and it is the
            # whole of DKV's system-RAM overhead over dense -- DKV's own
            # structures hold 0.00 GB of CPU tensors once loaded.
            #
            # device_map streams each shard straight to the GPU instead. The MPS
            # and quantized branches around this one already do it this way.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                local_files_only=self.local_files_only,
                **attn_impl_kwarg,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                quantization_config=quantization_config,
                use_safetensors=True,
                local_files_only=self.local_files_only,
                **attn_impl_kwarg,
            )

        self.model.eval()
        _clear_cpu_grad_state(self.model)
        _trim_python_heap()

        # ── Auto-detect standard 4-bit / 8-bit quantization ──
        is_quantized = False
        for name, module in self.model.named_modules():
            module_class = module.__class__.__name__.lower()
            if any(q_word in module_class for q_word in ["quant", "linear4bit", "linear8bit", "wqlinear", "bnb"]):
                is_quantized = True
                break
        
        if not is_quantized:
            for param in self.model.parameters():
                if param.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
                    is_quantized = True
                    break
        
        if is_quantized:
            print("[DKV] Auto-detected already quantized model. Skipping torchao post-quantization.")
        else:
            quant_type = config.get("quantization") or os.environ.get("DKV_QUANTIZATION")
            if quant_type in ["int8", "int4"]:
                if not _has_cuda() and not _has_mps():
                    print(f"[DKV] torchao {quant_type} quantization skipped on CPU.")
                elif _is_apple_silicon() and not _has_cuda():
                    if quant_type == "int4":
                        print("[DKV] int4 quantization on MPS is experimental.")
                    try:
                        from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
                        cfg = Int8WeightOnlyConfig() if quant_type == "int8" else Int4WeightOnlyConfig()
                        print(f"[DKV] Applying {quant_type} quantization via torchao on MPS...")
                        quantize_(self.model, cfg)
                        print("[DKV] torchao quantization applied successfully!")
                    except Exception as e:
                        print(f"[DKV] WARNING: torchao {quant_type} on MPS failed ({e}). Running in fp16.")
                else:
                    try:
                        from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
                        if quant_type == "int8":
                            quantize_(self.model, Int8WeightOnlyConfig())
                        elif quant_type == "int4":
                            quantize_(self.model, Int4WeightOnlyConfig())
                        print("[DKV] torchao quantization applied successfully!")
                    except Exception as e:
                        print(f"[DKV] WARNING: Failed to apply torchao weight quantization: {e}")

        self.num_layers = self.model.config.num_hidden_layers
        self.heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.heads
        if self.rank >= self.head_dim:
            old_rank = self.rank
            self.rank = self.head_dim // 2
            print(f"[DKV] WARNING: Capping SVD rank to {self.rank}")
        
        self.kv_heads = getattr(self.model.config, "num_key_value_heads", self.heads)
        self.serving_mode = config.get("serving_mode", "balanced")
        
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"[DKV] Model parameter count: {num_params / 1e6:.1f}M")
        except Exception:
            num_params = 1.5e9

        self.config = self.config or {}
        if "srl_k_min" not in self.config and "DKV_SRL_K_MIN" not in os.environ:
            if num_params < 1.0e9:
                self.config["srl_k_min"] = 10
            elif num_params < 3.0e9:
                self.config["srl_k_min"] = 15
            else:
                self.config["srl_k_min"] = 20

        if "srl_k_max" not in self.config and "DKV_SRL_K_MAX" not in os.environ:
            if num_params < 1.0e9:
                self.config["srl_k_max"] = 50
            elif num_params < 3.0e9:
                self.config["srl_k_max"] = 100
            else:
                self.config["srl_k_max"] = 200

        if "srl_threshold" not in self.config and "DKV_SRL_THRESHOLD" not in os.environ:
            if num_params < 1.0e9:
                self.config["srl_threshold"] = 25
            elif num_params < 3.0e9:
                self.config["srl_threshold"] = 40
            else:
                self.config["srl_threshold"] = 50

        self.manager = KVRuntimeManager(
            self.num_layers,
            self.kv_heads,
            self.head_dim,
            device=self.device,
            rank=self.rank,
            micro_block_size=self.micro_block_size,
            serving_mode=self.serving_mode,
            tokenizer=self.tokenizer,
            config=self.config,
        )
        self.manager.model_id = self.model_id
        self.manager.model = self.model
        
        # Load stop token IDs from model config if present
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            model_eos = getattr(self.model.generation_config, "eos_token_id", None)
            if isinstance(model_eos, list):
                self.stop_token_ids.update(model_eos)
            elif isinstance(model_eos, int):
                self.stop_token_ids.add(model_eos)

        apply_dkv_attention_patch(self.model, self.manager)

        # Tell the pool how many layers actually hold blocks.  The patch only
        # wraps full-attention layers, so on a hybrid model this is a fraction of
        # config.num_hidden_layers (6 of 24 on Qwen3.5-2B) and the pool would
        # otherwise reserve slots for layers that never allocate one.
        try:
            _n_attended = sum(
                1 for _l in self.model.model.layers
                if hasattr(getattr(_l, "self_attn", None), "_original_forward"))
            _pool = getattr(self.manager, "native_pool", None)
            if _pool is not None and _n_attended > 0:
                _pool.sizing_layers = _n_attended
                if _n_attended != self.num_layers:
                    print(f"[DKV] Pool sized for {_n_attended} attended layers "
                          f"(model has {self.num_layers}); hybrid model, the rest "
                          f"hold no compressed blocks.")
        except Exception:
            pass

        use_compile = "1" if self.manager.config.torch_compile else "0"
        if _is_apple_silicon():
            if use_compile == "auto":
                use_compile = "0"

        if use_compile == "0":
            print("[DKV] torch.compile disabled.")
        elif is_quantized and os.environ.get("DKV_FORCE_COMPILE_QUANTIZED", "0") != "1":
            # Quantized linears (bnb Linear4bit, AWQ WQLinear, ...) are opaque to
            # Inductor: it cannot fuse the dequantize step, so every FFN call
            # graph-breaks and each new prefill chunk shape triggers a fresh
            # compile.  On a 14B NF4 model that turned a 17.4s prefill into
            # 42.6s (first prompt) / 31.9s (recompile on the second) while
            # *lowering* decode throughput, because the compiled wrapper only
            # adds guard overhead around eager dequant kernels.
            #
            # The `high` preset sets torch_compile=True, which used to satisfy
            # `use_compile != "1"` and bypass this guard entirely.  Quantization
            # now wins over the preset: it is a property of the loaded weights,
            # not a user preference.  MLX has no equivalent step.
            print("[DKV] Quantized model detected — skipping torch.compile to avoid graph-break errors. "
                  "(Set DKV_FORCE_COMPILE_QUANTIZED=1 to override.)")
        else:
            # Pre-flight: verify the C++ compiler required by TorchInductor is available.
            # On Windows this is cl.exe (MSVC); on Linux/macOS it is gcc/clang.
            # On MPS we use 'aot_eager' which has no C++ compiler requirement.
            _compiler_ok = True
            import sys as _sys
            if _sys.platform == "win32":
                import shutil
                if shutil.which("cl") is None and use_compile != "1":
                    print("[DKV] torch.compile skipped — cl.exe (MSVC) not found in PATH. "
                          "Install Visual Studio Build Tools or set DKV_USE_TORCH_COMPILE=0 to silence this.")
                    _compiler_ok = False

            if _compiler_ok:
                _backend = _get_compile_backend()
                _mode    = _get_compile_mode()
                print(f"[DKV] Applying FFN-only layer torch.compile(dynamic=True, mode='{_mode}', backend='{_backend}') ...")
                try:
                    layers = getattr(self.model, "model", self.model).layers
                    compiled_count = 0
                    for layer in layers:
                        if hasattr(layer, "mlp") and layer.mlp is not None:
                            layer.mlp = torch.compile(
                                layer.mlp,
                                backend=_backend,
                                mode=_mode,
                                dynamic=True,
                                fullgraph=False,
                            )
                            compiled_count += 1
                    print(f"[DKV] FFN compilation applied successfully to {compiled_count} layers. First request will trigger JIT warmup.")
                except Exception as e:
                    print(f"[DKV] WARNING: FFN torch.compile failed ({e}). Running in eager mode.")

        # ── CUDA Graph Runner ────────────────────────────────────────────────
        # Created here so batch_engine.py can always find it via
        # getattr(wrapper, '_cuda_graph_runner', None).
        # On MPS/CPU: CUDAGraphDecodeRunner._capture_enabled=False so it's a no-op.
        if _HAS_CUDA_GRAPH_RUNNER:
            self._cuda_graph_runner = CUDAGraphDecodeRunner()
            _graph_enabled = bool(getattr(self._cuda_graph_runner, "capture_enabled", False))
            print(f"[DKV] CUDAGraphDecodeRunner initialized "
                  f"({'capture permitted — static ABI required' if _graph_enabled else 'capture disabled — eager mode'})")
        else:
            self._cuda_graph_runner = None

        if os.environ.get("DKV_SYNC_DEBUG", "0") == "1":
            _patch_tensor_sync_barriers()

        # ── Decode JIT pre-warm ──────────────────────────────────────────────
        # torch.compile() is lazy — Inductor only fires on the first REAL tensor
        # call.  Pre-trigger it here at load time using dummy tensors so neither
        # CLI users nor benchmark runs pay the 60-120s compile cost on their
        # first request.  Matches the behaviour of MLX's @mx.compile, which
        # compiles at definition time.
        #
        # Skip if DKV_JIT_SKIP_WARMUP=1 (useful for fast CI smoke tests that
        # don't exercise the CUDA decode path).
        if str(self.device).startswith("cuda") and os.environ.get("DKV_JIT_SKIP_WARMUP", "0") != "1":
            try:
                from native_core.sparse_decode.triton_fused_decode import warm_up_jit
                _cfg  = self.manager.config if hasattr(self, "manager") and self.manager else {}
                _dtype = torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else torch.float16
                # Read model's actual head counts from config for accurate dummy shapes
                _hf_cfg = getattr(self.model, "config", None)
                _H      = getattr(_hf_cfg, "num_attention_heads", 32)
                _kv_H   = getattr(_hf_cfg, "num_key_value_heads", _H)
                _D      = getattr(_hf_cfg, "head_dim",
                                  getattr(_hf_cfg, "hidden_size", 4096) // max(_H, 1))
                _R      = _cfg.get("rank", 16) if isinstance(_cfg, dict) else getattr(_cfg, "rank", 16)
                # Use the manager's actual block_size, not the config value.
                # The manager derives block_size independently (currently 64 by default);
                # the config dict may carry a different value that is ignored by the manager.
                _bs     = getattr(self.manager, "block_size", 64)

                # Determine all unique ranks that can be used across layers
                ranks_to_warm = {_R}
                num_layers = getattr(_hf_cfg, "num_hidden_layers", 28)
                try:
                    from native_core.kv_runtime_manager import get_layer_rank
                    _early_boost = getattr(_cfg, "early_layer_rank_boost", False)
                    _max_rank_early = getattr(_cfg, "max_rank_early", 0)
                    for l_idx in range(num_layers):
                        r = get_layer_rank(
                            l_idx, num_layers, _R,
                            early_boost=_early_boost, max_rank_early=_max_rank_early
                        )
                        ranks_to_warm.add(r)
                except Exception:
                    pass

                print(f"[DKV] Pre-warming decode JIT for ranks {sorted(list(ranks_to_warm))} (H={_H}, kv_H={_kv_H}, D={_D}, block_size={_bs}) ...", flush=True)
                for r in sorted(list(ranks_to_warm)):
                    warm_up_jit(device=self.device, dtype=_dtype, H=_H, kv_heads=_kv_H, D=_D, R=r, block_size=_bs)
            except Exception as _e:
                print(f"[DKV] WARNING: JIT pre-warm step failed ({_e}). "
                      "First decode request will trigger compilation.", flush=True)

        # ── Post-init memory cleanup ─────────────────────────────────────────
        # Fix 1B + 2.2 — run after everything is wired up so all temp objects are free.
        _clear_cpu_param_copies(self.model, self.device)   # audit stray CPU params + flush cache

        if self.device == "mps":
            # Cap MPS allocator fraction + start RSS pressure daemon.
            # Only set hard fraction cap if DKV_MPS_MEMORY_FRACTION env var is explicitly configured.
            # Otherwise, avoid setting it to prevent artificial allocator OOMs.
            mps_fraction_str = os.environ.get("DKV_MPS_MEMORY_FRACTION")
            if "mps_memory_fraction" in self.config:
                _configure_mps_memory(float(self.config["mps_memory_fraction"]))
            elif mps_fraction_str is not None:
                _configure_mps_memory(float(mps_fraction_str))
            else:
                _configure_mps_memory(None)

        # Fix 1B — trim Python heap to return model-loading arena to OS
        # (effective on macOS/Linux; no-op on Windows)
        _trim_python_heap()



    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def stop(self):
        """Cleanly release all resources and stop background worker threads."""
        if hasattr(self, "manager") and self.manager is not None:
            if hasattr(self.manager, "clear"):
                self.manager.clear()
            if hasattr(self.manager, "pager") and self.manager.pager is not None:
                if hasattr(self.manager.pager, "stop"):
                    self.manager.pager.stop()
            if hasattr(self.manager, "_compressor") and self.manager._compressor is not None:
                self.manager._compressor.stop()

        # Destroy CUDA graph runner if it was created
        if hasattr(self, "_cuda_graph_runner") and self._cuda_graph_runner is not None:
            try:
                self._cuda_graph_runner.invalidate()
            except Exception:
                pass


    @torch.no_grad()
    def _dkv_dump_attention_inputs(self, session_id, pend) -> None:
        """Checksum every input the replayed attention reads, per step.

        Probes from OUT here rather than inside the forward, because the forward
        does not execute under replay -- an in-forward probe prints nothing on
        exactly the steps in question, which is how two earlier faults stayed
        hidden.

        Read it by looking for a value that is FROZEN across replay steps and
        moves only on the periodic eager step. That is the signature of an input
        the graph is not re-reading, and it is what found the stale ingest refs.
        Addresses matter as much as contents: a buffer that is REALLOCATED leaves
        the graph reading the old address forever, and only ptr reveals that.
        """
        import sys as _s
        import torch as _t

        def _chk(t):
            if t is None:
                return "none"
            try:
                return f"0x{t.data_ptr():x}/{float(t.float().abs().sum()):.3f}"
            except Exception:                                    # noqa: BLE001
                return "err"

        mgr = self.manager
        ws = mgr.decode_workspace.get(session_id, {}) or {}
        L = 0
        k0 = (pend or {}).get(L)
        dwk = (ws.get("dense_workspace_k") or {}).get(L)
        dlen = (ws.get("dense_len_dev") or {}).get(L)
        routed = (ws.get("_routed_buf") or {}).get(L)
        remat = (ws.get("_remat_pin") or {}).get(L)
        cos = ws.get("rope_cos")
        # Compare against what CAPTURE bound, not the last eager forward.
        fwd = ((mgr.__dict__.get("_graph_fwd_ptrs")
                or mgr.__dict__.get("_fwd_ptrs") or {}).get(L)) or {}
        def _cmp(name, cur):
            f = fwd.get(name)
            if f is None or cur is None:
                return f"{name}:?"
            return f"{name}:{'SAME' if f == cur else 'DIFFERENT'}"
        print(
            f"[BIND] step={getattr(self, '_dkv_step_idx', -1):3d} "
            f"{_cmp('dense', None if dwk is None else dwk.data_ptr())} "
            f"{_cmp('bi', None if routed is None else routed[0].data_ptr())} "
            f"{_cmp('Km', None if remat is None else remat[0].data_ptr())}",
            flush=True, file=_s.stderr)
        print(
            f"[DUMP] step={getattr(self, '_dkv_step_idx', -1):3d} "
            f"replay={int(bool(getattr(self, '_dkv_last_was_replay', False)))} "
            f"| ingestK {_chk(None if k0 is None else k0[1])} "
            f"| window {_chk(dwk)} "
            f"| dlen {'none' if dlen is None else int(dlen.item())} "
            f"| routed {_chk(None if routed is None else routed[0])} "
            f"| remat {_chk(None if remat is None else remat[0])} "
            f"| cos {_chk(cos)}",
            flush=True, file=_s.stderr)
        # THE POOL ITSELF. Everything above is state the WRAPPER republishes
        # between forwards, so it advances under replay by construction. The
        # compressed pool is not: the gather reads pool.V_K / anchors_K and a
        # captured graph froze whatever it held at capture, so if these move
        # while a graph is live, replay is attending stale compressed content.
        _pool = getattr(mgr, "native_pool", None)
        _smgr = getattr(mgr, "_streaming_mgr", None)
        try:
            _gen = (_smgr._metadata_versions.get(session_id, {}).get(L)
                    if _smgr is not None else None)
        except Exception:                                        # noqa: BLE001
            _gen = None
        print(
            f"[POOL] step={getattr(self, '_dkv_step_idx', -1):3d} "
            f"gen {_gen} "
            f"| V_K {_chk(None if _pool is None else getattr(_pool, 'V_K', None))} "
            f"| ancK {_chk(None if _pool is None else getattr(_pool, 'anchors_K', None))} "
            f"| resK {_chk(None if _pool is None else getattr(_pool, 'residual_K_values', None))}",
            flush=True, file=_s.stderr)

    def _dkv_reset_graph_step(self) -> None:
        """Clear per-session graph state. Called from prefill, once per session."""
        self._dkv_step_idx = 0
        self._dkv_capture_giveup = False
        # CLEAR THE MODULE GLOBAL TOO. _MUTATION_OUT_ACTIVE lives on
        # runtime.dkv_attention, not on the wrapper, and it is republished per
        # STEP by _dkv_publish_mutation_out -- which only runs inside this
        # wrapper's decode loop. Anything that calls model() directly, as
        # ContinuousBatchEngine does, never republishes it and therefore inherits
        # whatever the last generate() on ANY wrapper left behind.
        #
        # That is a cross-session leak, and in a test process it is a cross-TEST
        # one: it is why test_formatting passes 6/6 alone and fails after the
        # engine tests have run. Prefill is the right place to clear it because
        # it is the one point every session passes through.
        try:
            import runtime.dkv_attention as _da_reset
            _da_reset._MUTATION_OUT_ACTIVE = False
        except Exception:                                        # noqa: BLE001
            pass
        self._dkv_primed = False
        # One-shot logs are per session too, otherwise the first session's
        # verdict is the only one ever reported and a later session that
        # declines for a DIFFERENT reason looks like it never happened.
        self._dkv_mo_off_logged = False
        self._graph_sel_logged = False

    def _dkv_routing_selective(self, session_id: str) -> bool:
        """Would a frozen routed set be WRONG for this session right now?

        Replay cannot re-run the Python router, so a captured routed set is
        frozen for the whole replay. When the router would have selected every
        block anyway -- n_blocks <= K -- freezing it is a no-op and replay is
        exact by construction. Above that line it drifts.

        Factored out of the capture block because two decisions need it and only
        one of them is about capturing. Capture asks once, when it is about to
        capture; mutation-out has to ask on EVERY step, because paying to defer
        the forward's mutation is only worth it if a graph is going to exist.

        Re-evaluated per step rather than cached per session on purpose: blocks
        keep getting compressed during decode, so a session can start at 15
        blocks with K=16 and cross the line mid-generation. Cached, it would keep
        deferring after the graph had already been declined -- paying the cost
        with nothing to show for it, which is the exact failure this gate exists
        to prevent.
        """
        ncomp, K = self._dkv_block_counts(session_id)
        return bool(ncomp is None or (K > 0 and ncomp > K))

    def _dkv_block_counts(self, session_id: str):
        """(compressed blocks, routing K) for this session, or (None, K) if unknown."""
        pool = getattr(self.manager, "native_pool", None)
        try:
            K = int(os.environ.get(
                "DKV_TOPK_BLOCKS",
                getattr(pool, "routing_topk_default", 16) or 16))
        except Exception:                                        # noqa: BLE001
            K = 16
        try:
            blocks = self.manager.get_streaming_blocks(session_id, 0) or []
            return sum(1 for b in blocks
                       if getattr(b, "state", "") == "COMPRESSED"), K
        except Exception:                                        # noqa: BLE001
            # Unknown -> caller treats this as selective. The conservative
            # direction is to DECLINE, because wrongly deferring costs
            # correctness under replay while wrongly not deferring costs speed.
            return None, K

    def _dkv_publish_mutation_out(self, session_id: str) -> None:
        """Set the effective mutation-out flag for the step about to run.

        --fastdc asks for mutation-out. Whether it is worth honouring depends on
        the session: it is a large win where a graph will be captured and a ~9%
        loss on wide models where the gate declines one. Deciding per session is
        what lets the flag be requested globally without that loss.
        """
        import runtime.dkv_attention as _da
        if not getattr(_da, "_MUTATION_OUT", False):
            _da._MUTATION_OUT_ACTIVE = False
            return

        # GIVE-UP BEATS EVERY OTHER REASON, including the manual override.
        # Selectivity is only ONE way to end up with no graph, and gating on it
        # alone was wrong: measured at 16k on Qwen2.5-1.5B, routing is
        # non-selective (15 blocks, K=16) so a selectivity-only gate keeps
        # deferring -- while capture is refused anyway for a completely
        # unrelated reason ("no decode cache supplied"), leaving --fastdc ~5%
        # SLOWER than default with byte-identical output:
        #
        #     fastdc off  11.84 / 11.97 tok/s      fastdc on  11.33 / 11.13
        #
        # So the question is not "would a graph be exact here" but "did one
        # actually take". After the first capture attempt the runner answers
        # that directly, and it covers every failure reason at once --
        # selectivity, missing decode cache, or a capture exception.
        if getattr(self, "_dkv_capture_giveup", False):
            _da._MUTATION_OUT_ACTIVE = False
            return
        ncomp, K = self._dkv_block_counts(session_id)

        # THREE cases, not two. "Not selective" is not the same question as
        # "mutation-out is safe here", and conflating them broke the 4k gates
        # (3/8, with output degenerating to "29dfulfulful"):
        #
        #   ncomp is None  state unreadable            -> off, conservatively
        #   ncomp == 0     DKV NOT ENGAGED at all      -> off. This is the case
        #                  that bit. A short prompt runs the dense/bypass
        #                  forward, which mutates its cache the ordinary way;
        #                  mutation-out defers nothing there, so claiming the
        #                  forward is read-only let capture skip a rollback it
        #                  genuinely needed.
        #   0 < ncomp <= K engaged and non-selective   -> ON, the win case
        #   ncomp > K      engaged and selective       -> off, frozen set drifts
        active = ncomp is not None and 0 < ncomp <= K
        if not active and not getattr(self, "_dkv_mo_off_logged", False):
            self._dkv_mo_off_logged = True
            _why = ("state unreadable" if ncomp is None
                    else "DKV routing is not engaged for this context"
                    if ncomp == 0 else
                    f"routing is selective ({ncomp} compressed blocks > K={K}), "
                    f"so a frozen routed set would drift")
            print(f"[DKV] mutation-out disabled for this session: {_why}. No "
                  f"routed graph will be captured, and deferring the forward's "
                  f"mutation would cost without buying anything.",
                  file=sys.stderr, flush=True)
        _da._MUTATION_OUT_ACTIVE = active

    def _dkv_apply_pending_mutation(self, session_id: str) -> None:
        """Perform the decode step's state mutation OUTSIDE the forward.

        Only active under DKV_GRAPH_MUTATION_OUT. The forward has been made
        read-only with respect to KV state so a CUDA graph can replay it; the
        two mutations it used to do are performed here instead, between forwards,
        where Python actually runs.

        Order matters and mirrors what the forward used to do:
          1. ingest the token the forward just produced, from the K/V references
             the forward stashed. Under replay those references point into the
             graph's fixed-address buffers and hold the values this replay
             produced, which is what makes this work at all.
          2. rebuild each layer's dense window and publish its length as a DEVICE
             tensor, so the next forward can mask by it instead of slicing with a
             host int a captured graph would bake in.
        """
        import runtime.dkv_attention as _da
        # The EFFECTIVE flag, not the requested one. This must agree with what
        # the forward that just ran actually did: if the forward did not defer
        # (mutation-out inactive for this session) there is nothing to drain,
        # and reading the requested flag here would have this run against a
        # stale stash on exactly the sessions where the gate turned it off.
        if not getattr(_da, "_MUTATION_OUT_ACTIVE", False):
            return
        mgr = self.manager
        # After a REPLAY the eager stash is stale by construction (see the
        # snapshot at capture). Prefer the capture-time refs whenever the last
        # step was replayed.
        if getattr(self, "_dkv_last_was_replay", False):
            pend = mgr.__dict__.get("_graph_ingest_refs") or {}
        else:
            pend = mgr.__dict__.get("_pending_ingest") or {}

        if not pend:
            _owed = False
            try:
                from native_core.sparse_decode.triton_fused_decode import (
                    pool_stores_rotated_k as _psr_e)
                _owed = (not _psr_e()) and not (
                    (mgr.decode_workspace.get(session_id, {}) or {})
                    .get("dense_rot_dev"))
            except Exception:                                    # noqa: BLE001
                _owed = False
            if not _owed:
                return
        import torch as _t
        # Assert the invariant this branch should have carried from the start:
        # layer 0's block must grow by EXACTLY ONE token per generate step. A
        # double-ingest and a missed ingest both produce fluent-but-wrong text and
        # are indistinguishable from the output; they are trivially
        # distinguishable here.
        _dbg = os.environ.get("DKV_GRAPH_DEBUG_PTR") == "1"
        _before = None
        if _dbg:
            try:
                _b0 = [b for b in (mgr.get_streaming_blocks(session_id, 0) or [])
                       if b.state == "ACCUMULATING"]
                _before = sum(int(b.active_k.shape[2]) for b in _b0
                              if b.active_k is not None)
            except Exception:                                    # noqa: BLE001
                _before = None
        for layer_idx, (sid, k, v) in list(pend.items()):
            try:
                mgr.ingest_streaming(sid, layer_idx, k, v)
            except Exception:                                    # noqa: BLE001
                pass
        if _dbg and _before is not None:
            import sys as _s3
            try:
                _b1 = [b for b in (mgr.get_streaming_blocks(session_id, 0) or [])
                       if b.state == "ACCUMULATING"]
                _after = sum(int(b.active_k.shape[2]) for b in _b1
                             if b.active_k is not None)
                _d = _after - _before
                _src = "graphrefs" if getattr(self, "_dkv_last_was_replay", False) else "pending"
                _k0v = (pend or {}).get(0)
                _ksum = "-" if _k0v is None else f"{float(_k0v[1].float().abs().sum()):.2f}"
                print(f"[INGEST] step={getattr(self,'_dkv_step_idx',-1):3d} "
                      f"replay={int(bool(getattr(self,'_dkv_last_was_replay',False)))} "
                      f"src={_src:9s} Ksum={_ksum} "
                      f"L0 {_before}->{_after} delta={_d}"
                      f"{'  <-- EXPECTED 1' if _d != 1 else ''}",
                      flush=True, file=_s3.stderr)
            except Exception:                                    # noqa: BLE001
                pass
        # DO NOT clear. The entries are REFERENCES into the graph's fixed-address
        # buffers, and the line that records them lives inside the forward -- which
        # replay does not execute. Clearing them made every token after the first
        # replay ingest nothing at all, silently freezing the KV store while the
        # text stayed fluent. Keeping them is the whole point: each replay
        # rewrites those same addresses, so re-reading them here ingests the token
        # this step actually produced. Eager steps simply overwrite the same keys.

        ws = mgr.decode_workspace.setdefault(session_id, {})
        lens = ws.setdefault("dense_len_dev", {})
        for layer_idx in range(mgr.num_layers):
            try:
                blocks = mgr.get_streaming_blocks(session_id, layer_idx)
                dense_blocks = [b for b in (blocks or []) if b.state == "ACCUMULATING"]
                if not dense_blocks:
                    continue
                # Use the dtype the WORKSPACE already has, not the model's.
                # assemble_dense_window_kv reallocates when the dtype differs,
                # which would rebind dense_workspace_k to a NEW tensor and leave
                # a captured graph reading the old address forever.
                # FORCE the copy. assemble_dense_window_kv only writes a block
                # when blk.dirty is set, or when its growth check sees the cached
                # per-block active_len differ from the current one. Both of those
                # are bookkeeping the FORWARD maintains, and under replay the
                # forward does not run -- so the function walked the blocks,
                # recomputed a LARGER dlen from the live views, and copied
                # nothing. Measured: each eager step added ~6400 to the window
                # abs-sum (one token's |K|), each replay step added exactly 0
                # while dlen still incremented.
                #
                # Marking the tail block dirty here is the caller taking
                # responsibility for the invariant instead of inheriting it from
                # a path that no longer executes.
                for _b in dense_blocks:
                    _b.dirty = True
                _existing = (ws.get("dense_workspace_k") or {}).get(layer_idx)
                _dt = _existing.dtype if _existing is not None else self.model.dtype
                _ptr_before = None if _existing is None else _existing.data_ptr()
                _dk, _dv, dlen, _trimmed = mgr.assemble_dense_window_kv(
                    session_id, layer_idx, dense_blocks, _dt)
                # PUBLISH THE TRIMMED BLOCK LIST. _remat_attend needs it to
                # rotate the dense window when the pool is unrotated, and on this
                # path it is the wrapper -- not the forward -- that assembles the
                # window, so the forward has no other way to reach it.
                #
                # It must be the TRIMMED list the assembler returned, not the one
                # passed in: the assembler drops blocks to fit the workspace, and
                # the positions have to describe what was actually written.
                #
                # HONEST NOTE ON WHAT THIS DID AND DID NOT FIX. It was added to
                # explain --fastdc diverging from eager at 16k on an unrotated
                # pool (md5 0fec68e1 eager against 566e1b26 replayed, with
                # DKV_DETERMINISTIC=1 on both), on the theory that remat was
                # declining here for want of positions and falling back to the
                # Triton kernel. Plumbing the list through changed NOTHING --
                # both md5s are unchanged -- so that theory is wrong and the
                # divergence lies in the replay itself, not in this path's
                # attention dispatch. Kept because it is still correct (this path
                # can now serve remat on an unrotated pool at all) but it is
                # NOT the explanation for the --fastdc divergence.
                ws.setdefault("dense_blocks_trimmed", {})[layer_idx] = _trimmed
                try:
                    from native_core.sparse_decode.triton_fused_decode import (
                        pool_stores_rotated_k as _psr_w,
                        _partial_rope_apply as _pra_w,
                    )
                    if _dk is not None and dlen and not _psr_w():
                        _pw = []
                        for _b in (_trimmed or []):
                            _pw.extend(getattr(_b, "token_indices", ()) or ())
                        _vw = min(len(_pw), int(dlen))
                        if _vw > 0:
                            import runtime.dkv_attention as _da_w
                            _cw, _sw = _da_w._history_cos_sin(
                                self.model, _dk, int(max(_pw[:_vw])) + 1, _dk.device)
                            _rotd = ws.setdefault("dense_rot_dev", {})
                            _bufw = _rotd.get(layer_idx)
                            if (_bufw is None or _bufw.shape != _dk.shape
                                    or _bufw.dtype != _dk.dtype):
                                # ZEROS, not empty: only the valid prefix is
                                # written below, so the tail must start defined
                                # and stay that way. The assembler zeroes its own
                                # tail for the same reason.
                                _bufw = torch.zeros_like(_dk)
                                _rotd[layer_idx] = _bufw
                            # INCREMENTAL. A row's rotation depends only on its
                            # own absolute position, which never changes, so a
                            # row rotated on an earlier step is still correct on
                            # this one. Only the rows APPENDED since last step
                            # need doing -- normally one.
                            #
                            # Rotating the whole valid prefix every step was
                            # ~1460 rows per layer per step and showed up as
                            # ~50 ms of extra elementwise work in the 16k
                            # profile. Copying the whole 3072-row workspace on
                            # top of that was worse again. Both are gone.
                            #
                            # The layout is only stable while the BLOCK SET is:
                            # the assembler repacks from row 0 when a block
                            # leaves (it protects block 0 and drops the
                            # second-oldest), which moves every row. Its own
                            # signature says when that happened, so key the
                            # incremental state on it and rebuild in full when
                            # it changes.
                            _sig = tuple(getattr(b, "anchor_idx", -1)
                                         for b in (_trimmed or []))
                            # KEY COLLISION, FIXED. This used to be
                            # "dense_rot_state", which is ALSO the key
                            # dkv_attention.py's combined branch uses for a
                            # different cache with an incompatible value: that
                            # one stores a dict {version, anchors, lengths,
                            # rot}, this one a tuple (sig, valid_len). Whichever
                            # wrote second poisoned the other, and the forward
                            # -- unlike this block, which is inside a try --
                            # dereferenced it unguarded and died with
                            # "'tuple' object has no attribute 'get'".
                            #
                            # Only reachable when BOTH are live: the combined
                            # branch needs DKV_SPARSE_BIAS unset or "0.0" (the
                            # LIBRARY DEFAULT) and this pre-rotation needs the
                            # mutation-out path. BEST_DECODE_DEFAULTS sets
                            # DKV_SPARSE_BIAS=auto, which takes the production
                            # branch instead -- so everything going through the
                            # serving defaults, validate_cuda_dkv.py included,
                            # missed it. Same blind spot as the combined-window
                            # frame defect.
                            _rstate = ws.setdefault("dense_prerot_state", {})
                            _prev_sig, _prev_len = _rstate.get(layer_idx,
                                                               (None, 0))
                            _from = _prev_len if (_prev_sig == _sig
                                                  and _prev_len <= _vw) else 0
                            _rstate[layer_idx] = (_sig, _vw)
                            if _from < _vw:
                                _dpw = torch.as_tensor(_pw[_from:_vw],
                                                       dtype=torch.long,
                                                       device=_dk.device)
                                _cd = _cw[0, _dpw.clamp(max=_cw.shape[1] - 1)].unsqueeze(0).unsqueeze(1)
                                _sd = _sw[0, _dpw.clamp(max=_sw.shape[1] - 1)].unsqueeze(0).unsqueeze(1)
                                _bufw[:, :, _from:_vw] = _pra_w(
                                    _dk[:, :, _from:_vw], _cd.to(_dk.dtype),
                                    _sd.to(_dk.dtype))
                except Exception as _rot_err:                    # noqa: BLE001
                    (ws.get("dense_rot_dev") or {}).pop(layer_idx, None)
                    if not getattr(self, "_rot_pub_err_logged", False):
                        self._rot_pub_err_logged = True
                        print(f"[DKV] dense-window pre-rotation failed "
                              f"({type(_rot_err).__name__}: {_rot_err})",
                              file=sys.stderr, flush=True)
                if (os.environ.get("DKV_GRAPH_DEBUG_PTR") == "1"
                        and layer_idx == 0 and _ptr_before is not None):
                    import sys as _s2
                    _after = (ws.get("dense_workspace_k") or {}).get(layer_idx)
                    print(f"[DT] L0 dt={_dt} model_dt={self.model.dtype} "
                          f"ptr {hex(_ptr_before)} -> "
                          f"{'same' if _after is not None and _after.data_ptr() == _ptr_before else 'REALLOC'} "
                          f"dirty={[getattr(b,'dirty',None) for b in dense_blocks][:3]} "
                          f"dlen={dlen}", flush=True, file=_s2.stderr)
                cur = lens.get(layer_idx)
                if cur is None:
                    lens[layer_idx] = _t.tensor(int(dlen), device=_dk.device,
                                                dtype=_t.long)
                else:
                    # in place: the mask built in the forward reads THIS tensor,
                    # and rebinding it would leave a captured graph reading the
                    # old address forever.
                    cur.fill_(int(dlen))
            except Exception:                                    # noqa: BLE001
                continue

        # AFTER the ingest and the reassembly, not before. Printing at the top
        # reported the window as it stood BEFORE this step's assemble, which is a
        # one-step lag -- and a lag reads exactly like a freeze if you are looking
        # for one. Every conclusion drawn from the earlier placement has to be
        # re-checked against this.
        if os.environ.get("DKV_GRAPH_DEBUG_PTR") == "1":
            self._dkv_dump_attention_inputs(session_id, pend)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        query_text: Optional[str] = None,
    ):
        session_id = self.active_session or "default"

        # O(1) Smart Prefix Check: check if the session already has resident KV cache.
        # If so, mark the cached length so prefill is incremental (avoiding O(N) re-prefill of history).
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_ids = inputs.input_ids[0].tolist()

        # Entity-binding hint: the actual question span inside the prompt.
        # Used by the factual store to bind decode to the queried entity (no-op
        # unless the factual store is enabled).
        #
        # Identical logic to mlx_dkv_wrapper.generate(): position-agnostic —
        # works whether the question is at the start, middle, or end of the
        # context. _pending_query is consumed by manager.finalize_srl_index().
        #
        # Priority:
        #   1. Explicit query_text from caller (highest precision).
        #   2. Auto-extracted from _last_messages (set by API gateway).
        #   3. Full prompt fallback (safe — IDF filters downstream).
        #
        # NO LONGER GATED ON THE FACTUAL STORE. `_pending_query` is consumed by
        # finalize_srl_index to set `current_query_tokens`, which is read as a
        # LEXICAL QUERY by query_router's lexical lookup, the decode-time
        # query_toks set, and the learned router's `lex` feature -- none of
        # which are factual-store features. Behind `_factual_enabled` (off by
        # default) the pin was never filled on the default path, so those three
        # consumers silently fell back to "the question is the entire prompt",
        # whose IDF is ~uniform and discriminates nothing. MLX has never gated
        # it this way.
        if True:
            try:
                if query_text:
                    _q_ids = self.tokenizer.encode(
                        query_text, add_special_tokens=False
                    )
                    if hasattr(_q_ids, "tolist"):
                        _q_ids = _q_ids.tolist()
                else:
                    _messages = getattr(self, "_last_messages", None)
                    _q_ids = _extract_query_token_ids(
                        self.tokenizer, prompt_ids, _messages
                    )
                if _q_ids:
                    self.manager._pending_query[session_id] = _q_ids
            except Exception:
                pass

        # ── Instruction-pinning: compute answer-candidate blocks pre-prefill ──
        # On HF/CUDA the sparse prefill is not yet ported (MLX-only),
        # but the pinned block list is stored so the decode-time residual
        # router can use it as a block-priority hint.
        #
        # Same fix as MLX wrapper: always extract query span via chat-template
        # rather than falling back to full prompt_ids (which has IDF ~1.0 for
        # everything and would pin nothing useful).
        if getattr(self.manager, '_sp_instr_pin', False):
            try:
                _q_ids_pin = getattr(self.manager, '_pending_query', {}).get(session_id)
                if not _q_ids_pin:
                    _messages = getattr(self, "_last_messages", None)
                    _q_ids_pin = _extract_query_token_ids(
                        self.tokenizer, prompt_ids, _messages
                    )
                _pinned = _pinned_blocks_from_prompt(
                    prompt_ids,
                    _q_ids_pin,
                    block_size=getattr(self.manager, 'block_size', 256),
                    stop_ids=getattr(self.manager, '_stop_token_ids', None),
                    idf_threshold=getattr(self.manager, '_sp_pin_idf', 3.0),
                    max_pinned_blocks=getattr(self.manager, '_sp_pin_max', 4),
                )
                self.manager._sp_pinned_blocks[session_id] = tuple(_pinned)
                if _pinned:
                    print(f"[DKV] Instruction-pinning: {len(_pinned)} block(s) pinned "
                          f"→ {_pinned}", flush=True)
            except Exception:
                pass


        cached_len = 0
        if hasattr(self.manager, "get_session_sequence_length"):
            seq_len = self.manager.get_session_sequence_length(session_id)
            if seq_len > 0 and seq_len < len(prompt_ids):
                stored_ids = getattr(self, "_session_token_ids", {}).setdefault(session_id, [])
                if len(stored_ids) >= seq_len and prompt_ids[:seq_len] == stored_ids[:seq_len]:
                    cached_len = seq_len
                    print(f"[DKV Wrapper] Found cached history for session {session_id}: length {cached_len} tokens. Reusing KV cache!")
                    
        if cached_len == 0:
            self.manager.clear_session(session_id)
            if not hasattr(self, "_session_token_ids"):
                self._session_token_ids = {}
            self._session_token_ids[session_id] = []
            new_prompt_ids = prompt_ids
        else:
            new_prompt_ids = prompt_ids[cached_len:]

        input_ids = torch.tensor([new_prompt_ids], device=self.device)
        prefill_len = input_ids.shape[1]
        generated = prompt_ids.copy()

        self.manager.init_session(session_id, prefill_len=cached_len + prefill_len)
        if hasattr(self.manager, "register_prefill_tokens"):
            self.manager.register_prefill_tokens(session_id, torch.tensor(new_prompt_ids, dtype=torch.long))
        self.model._dkv_session_ids = [session_id]

        # Invalidate CUDA graph runner — new prefill changes pool layout
        if hasattr(self, "_cuda_graph_runner") and self._cuda_graph_runner is not None:
            self._cuda_graph_runner.invalidate()
        # …and clear the per-session graph decisions with it. _dkv_capture_giveup
        # is sticky by design WITHIN a session, but it was being set on a wrapper
        # that outlives the session: _dkv_reset_graph_step(), which is where it
        # was cleared, is defined and never called. One context that cannot
        # capture -- a short prompt, or a long one where routing is selective --
        # therefore disabled mutation-out permanently for every LATER generate()
        # on the same wrapper, which in a server is every subsequent request.
        self._dkv_reset_graph_step()

        # ── Chunked prefill ──────────────────────────────────────────────────
        # Process the prompt in aligned chunks. Blocks remain dense through the
        # final prefill forward so later chunks see exact raw history; SVD is
        # published once at the prefill→decode boundary. This:
          #   1. Eliminates the O(N²) attention VRAM spike from one giant forward.
          #   2. Preserves exact causal prefill semantics like the MLX path.
          #   3. Avoids repeated partial-block SVD launches.
        PREFILL_CHUNK = getattr(self.manager, "config", None) and self.manager.config.prefill_chunk_size or 512
        # Keep CUDA chunk boundaries aligned with the streaming block layout.
        # Otherwise, e.g. 1024 tokens with 256 active tokens per block creates
        # a 252-token partial block on every chunk.  MLX keeps a single dense
        # tail; matching that shape avoids repeated tiny rSVD launches and
        # prevents old partial blocks from becoming an artificial barrier.
        if isinstance(self.device, torch.device):
            _is_cuda_device = self.device.type == "cuda"
        else:
            _is_cuda_device = str(self.device).startswith("cuda")
        if _is_cuda_device and hasattr(self.manager, "get_session_micro_block_size"):
            _mbs = self.manager.get_session_micro_block_size(session_id)
            _block_capacity = max(2, int(_mbs) + 1)
            PREFILL_CHUNK = ((PREFILL_CHUNK + _block_capacity - 1) // _block_capacity) * _block_capacity
        new_ids_list = new_prompt_ids
        total_new = len(new_ids_list)
        outputs = None
        import time as _time

        # ── Explicit prefill cache, threaded through every chunk ────────────
        # Without this, each chunk call below passed no `past_key_values` at
        # all, so HF auto-creates a FRESH empty cache per chunk -- DKV-patched
        # (self_attn) layers don't care (they manage their own state
        # externally), but hybrid architectures (Qwen3-Next/Qwen3.5-style)
        # also have non-attention layers (linear/gated-delta-net) DKV never
        # touches, which DO rely on this cache to carry their recurrent state
        # across chunks -- losing it silently corrupts any prompt longer than
        # one PREFILL_CHUNK. MLX's equivalent (_get_or_create_prefill_cache)
        # already builds one cache up front and threads it through every
        # chunk; this mirrors that.
        #
        # DKV_PREFILL_CACHE_BITS (MLX parity: same env var name/values):
        #   16 (default) -> plain DynamicCache
        #   8 or 4       -> QuantizedCache (needs the `hqq` or `quanto` package;
        #                   DKV_PREFILL_CACHE_BACKEND selects which, default
        #                   "hqq"). Falls back to DynamicCache with a one-time
        #                   warning if the backend package isn't installed.
        from transformers.cache_utils import DynamicCache
        try:
            _prefill_cache_bits = int(os.environ.get("DKV_PREFILL_CACHE_BITS", "16"))
        except ValueError:
            _prefill_cache_bits = 16
        prefill_cache = None
        if _prefill_cache_bits in (4, 8):
            try:
                from transformers.cache_utils import QuantizedCache
                _backend = os.environ.get("DKV_PREFILL_CACHE_BACKEND", "hqq")
                prefill_cache = QuantizedCache(
                    backend=_backend,
                    config=self.model.config,
                    nbits=_prefill_cache_bits,
                )
            except Exception as _qe:
                print(f"[DKV] WARNING: quantized prefill cache unavailable ({_qe}) "
                      f"-- install the '{os.environ.get('DKV_PREFILL_CACHE_BACKEND', 'hqq')}' "
                      "package, or set DKV_PREFILL_CACHE_BITS=16. Falling back to fp16 cache.")
        if prefill_cache is None:
            # config= is required for hybrid architectures (Qwen3-Next/Qwen3.5-
            # style): DynamicCache uses it to pre-size self.layers with the
            # right per-layer cache type (linear-attention conv/recurrent state
            # vs standard KV) up front. Without it, a bare DynamicCache() only
            # grows generically as if every layer were plain attention, and
            # crashes ("list index out of range") the moment a linear-attention
            # layer tries to update its conv state.
            prefill_cache = DynamicCache(config=self.model.config)

        # Pre-allocate reusable buffers for the entire prefill loop.
        # Using torch.tensor([chunk]) inside the loop creates a new Python list
        # object + a new Tensor object + new Storage on every chunk.
        # For 32 chunks (8200 tokens / 256 chunk_size), that is 32 × N allocations
        # per layer that the Python malloc arena never returns to the OS.
        # Using a single pre-allocated buffer filled in-place eliminates this.
        import gc as _gc
        _prefill_buf = torch.zeros((1, PREFILL_CHUNK), dtype=torch.long)
        _pos_buf     = torch.zeros((1, PREFILL_CHUNK), dtype=torch.long)

        for chunk_idx, chunk_start in enumerate(range(0, total_new, PREFILL_CHUNK)):
            chunk_end = min(chunk_start + PREFILL_CHUNK, total_new)
            chunk = new_ids_list[chunk_start:chunk_end]
            clen = chunk_end - chunk_start
            abs_start = cached_len + chunk_start
            is_last_chunk = (chunk_end >= total_new)

            # Re-use the pre-allocated buffer — fill in-place, no new Python objects
            _prefill_buf[0, :clen] = torch.as_tensor(chunk, dtype=torch.long)
            chunk_tensor = _prefill_buf[:, :clen].to(self.device, non_blocking=True)
            _pos_buf[0, :clen] = torch.arange(abs_start, abs_start + clen, dtype=torch.long)
            pos_tensor = _pos_buf[:, :clen].to(self.device, non_blocking=True)

            # Finalize any completed CPU background compressions from the previous chunk
            if hasattr(self.manager, "finalize_compressed_blocks"):
                self.manager.finalize_compressed_blocks()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=chunk_tensor,
                    position_ids=pos_tensor,
                    past_key_values=prefill_cache,
                    use_cache=True,
                )

            # Flush Metal command buffers every 4 chunks and run GC so freed
            # Python heap pages are returned to the OS (prevents 2 GB swap growth).
            if self.device == "mps":
                if is_last_chunk or (chunk_idx % 4 == 3):
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                        _gc.collect()  # return freed malloc arenas to the OS
                    except Exception:
                        pass

            # Keep this compatibility call; it only drains completed work and
            # does not publish new SVD blocks during prefill.
            if hasattr(self.manager, "compress_prefill_kv"):
                self.manager.compress_prefill_kv(session_id)

        # Release pre-allocated buffers
        del _prefill_buf, _pos_buf

        # Snapshot the final prefill logits before compression/SRL finalization
        # can allocate or mutate CUDA workspaces. MLX evaluates the output
        # before changing its dense/compressed cache state; keep the same
        # ordering on CUDA.
        prefill_logits = outputs.logits[:, -1, :].clone()

        # ── Post-prefill compression barrier ────────────────────────────────
        # Trigger SVD compression for all deferred prefill blocks
        if hasattr(self.manager, "compress_deferred_prefill_blocks"):
            self.manager.compress_deferred_prefill_blocks(session_id)

        # Drain all background SVD results to the native pool before decode starts.
        if hasattr(self.manager, "finalize_compressed_blocks"):
            _barrier_deadline = _time.monotonic() + 30.0
            while _time.monotonic() < _barrier_deadline:
                self.manager.finalize_compressed_blocks()
                pending = getattr(self.manager, "_pending_cpu_blocks", 0)
                if pending <= 0:
                    break
                _time.sleep(0.002)
            if self.device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        # Build SRL index once compression is completed
        if hasattr(self.manager, "finalize_srl_index"):
            self.manager.finalize_srl_index(session_id, cached_len=cached_len)

        srl_state = getattr(self.manager, "_session_srl", {}).get(session_id)
        if srl_state is not None:
            srl_state.vsl_active_candidates = []
            srl_state.vsl_consecutive_helpers = 0
            srl_state.factual_anchor_q = None
            srl_state.current_entity_id = -1
            srl_state.dual_entity_mode = False
            srl_state.dual_entity_ids = []

        past_kv = outputs.past_key_values
        logits = prefill_logits  # [1, vocab]

        # CRITICAL FIX: track the absolute sequence position for each decode step.
        cur_pos = cached_len + prefill_len

        # Pre-allocate position cache to avoid slow GPU allocations in the loop
        max_total_len = cur_pos + max_new_tokens + 10
        pos_cache = torch.arange(max_total_len, dtype=torch.long, device=self.device)

        # ── Context-Aware Decoding (CAD) setup — PyTorch/CUDA port ──────────────
        # Mirrors mlx_dkv_wrapper.generate: contrast full-context logits against
        # a PRIOR-only stream (question, no document) to pull the decoder off its
        # pretrained prior onto the document's relation:
        #     logits ← (1+α)·logits_full − α·logits_prior
        # The prior runs as its own short DKV session (own past_kv), advanced by
        # plain eager forward (device-agnostic — no CUDA/MPS graph capture needed).
        # Gated by DKV_CAD_ALPHA (0 = off); DKV_CAD_MAX_STEPS caps it to the
        # first N tokens. Runs identically on CUDA / MPS / CPU.
        try:
            _cad_alpha = float(os.environ.get("DKV_CAD_ALPHA", "0"))
        except ValueError:
            _cad_alpha = 0.0
        try:
            _cad_max_steps = int(os.environ.get("DKV_CAD_MAX_STEPS", "0"))
        except ValueError:
            _cad_max_steps = 0
        _cad_on = _cad_alpha > 0.0 and bool(query_text)
        _cad_prior_logits = None
        _cad_sid = None
        _cad_past = None
        _cad_pos = 0
        _cad_step = 0
        if _cad_on:
            try:
                _pri_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": query_text}],
                    tokenize=False, add_generation_prompt=True)
                _pri_ids = self.tokenizer(_pri_text, return_tensors="pt").input_ids.to(self.device)
            except Exception:
                _pri_ids = self.tokenizer(query_text, return_tensors="pt").input_ids.to(self.device)
            _cad_sid = session_id + "::cadprior"
            try:
                self.manager.clear_session(_cad_sid)
                self.manager.init_session(_cad_sid, prefill_len=_pri_ids.shape[1])
                if hasattr(self.manager, "register_prefill_tokens"):
                    self.manager.register_prefill_tokens(_cad_sid, _pri_ids[0].detach().cpu())
                self.model._dkv_session_ids = [_cad_sid]
                _pp = torch.arange(_pri_ids.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _po = self.model(input_ids=_pri_ids, position_ids=_pp, use_cache=True)
                _cad_prior_logits = _po.logits[:, -1, :]
                _cad_past = _po.past_key_values
                _cad_pos = _pri_ids.shape[1]
                if hasattr(self.manager, "compress_deferred_prefill_blocks"):
                    self.manager.compress_deferred_prefill_blocks(_cad_sid)
            except Exception as _e:
                print(f"[DKV HF CAD] disabled (prior prefill failed: {_e})")
                _cad_on = False
            finally:
                self.model._dkv_session_ids = [session_id]

        sfa_active = False

        for _ in range(max_new_tokens):
            # Context-Aware Decoding: extrapolate away from the prior-only stream
            # BEFORE rep-penalty / factual bias / sampling.
            if _cad_on and _cad_prior_logits is not None:
                logits = (1.0 + _cad_alpha) * logits - _cad_alpha * _cad_prior_logits
            # ── Repetition-loop detection (mirrors batch_engine.py / mlx_dkv_wrapper.py Fix 2) ──────
            # Detect tight token-level loops every 10 new tokens.
            # On detection, widen the penalty window and boost the strength.
            # After 40 tokens without recovery, force-stop generation.
            _new_tokens = generated[len(prompt_ids):]  # tokens produced in this call
            _n_new = len(_new_tokens)
            _loop_detected = getattr(self, "_hf_loop_detected", False)
            _loop_idx = getattr(self, "_hf_loop_idx", None)

            if not _loop_detected and _n_new >= 30 and _n_new % 10 == 0:
                _window = _new_tokens[-80:]
                _ng = 5
                if len(_window) >= _ng + 1:
                    _ngrams = [tuple(_window[i:i + _ng]) for i in range(len(_window) - _ng + 1)]
                    _top = Counter(_ngrams).most_common(1)[0][1]
                    if _top / len(_ngrams) >= 0.35:
                        _loop_detected = True
                        self._hf_loop_detected = True
                        self._hf_loop_idx = _n_new
                        print(
                            f"[DKV HF] WARNING: repetition loop detected at token "
                            f"{_n_new}. Escalating penalty window to 256 tokens and strength to 1.3x.",
                            file=sys.stderr
                        )

            if _loop_detected:
                if _loop_idx is None:
                    self._hf_loop_idx = _n_new
                elif _n_new - _loop_idx >= 40:
                    print(
                        "[DKV HF] WARNING: repetition loop persisted for 40 tokens "
                        "after detection — forcing EOS.",
                        file=sys.stderr
                    )
                    break

            # Repetition penalty (widened window when a loop is active)
            _pen_window = 256 if _loop_detected else 64
            _pen_val = max(repetition_penalty, 1.3) if _loop_detected else repetition_penalty
            if _pen_val != 1.0:
                # Numeric exemption (2026-07-13): digits carry semantics, not
                # fluency — penalizing them corrupts faithful reproduction of
                # numeric content (measured: CLI 12k table reproduction emitted
                # header + EMPTY cells, every digit argmax-suppressed at the
                # default 1.15, while the raw-argmax probe read the identical
                # compressed state 6/6). Suspended during loop recovery so the
                # escalated penalty still breaks digit loops. Mirrors
                # _filter_penalty_ids (batch_engine.py) and rep_exempt_cache
                # (native main.cpp). DKV_REP_PENALTY_PROTECT_NUMERIC=0 restores.
                _protect_numeric = (not _loop_detected and
                                    os.environ.get("DKV_REP_PENALTY_PROTECT_NUMERIC", "1") == "1")
                # Table-line suspension (mirror of batch_engine._in_table_line):
                # while the current output line (plus the line above — row
                # starts count) is table-like, suspend the penalty entirely;
                # verbatim table rows can't survive ANY penalized token
                # (measured: empty '| | | |' cells with digits exempt but
                # glue penalized). Loop recovery overrides.
                if _protect_numeric and generated:
                    if not hasattr(self, "_rep_decode_strs"):
                        self._rep_decode_strs = {}
                    _seps = _nums = _nl = _n = 0
                    for _tid in reversed(generated):
                        _n += 1
                        if _n > 64:
                            break
                        _s = self._rep_decode_strs.get(_tid)
                        if _s is None:
                            _s = self._rep_decode_strs[_tid] = self.tokenizer.decode([_tid])
                        if "\n" in _s:
                            _nl += 1
                            if _nl >= 2:
                                break
                            continue
                        _sc = _s.strip()
                        if _sc in ("|", "&"):
                            _seps += 1
                            _nums += 1
                        elif any(c.isdigit() for c in _sc):
                            _nums += 1
                        if _seps >= 2 or _nums >= 3:
                            _pen_val = 1.0
                            break
                if not hasattr(self, "_rep_exempt_tokens"):
                    self._rep_exempt_tokens = {}
                for tok_id in set(generated[-_pen_window:]):
                    if tok_id < logits.shape[-1]:
                        # Skip punctuation/newlines/whitespace to avoid suppressing lists/bullets/formatting
                        is_alnum = self._alphanumeric_tokens.get(tok_id)
                        if is_alnum is None:
                            tok_text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                            is_alnum = any(c.isalnum() for c in tok_text)
                            self._alphanumeric_tokens[tok_id] = is_alnum

                        if not is_alnum:
                            continue

                        if _protect_numeric:
                            _ex = self._rep_exempt_tokens.get(tok_id)
                            if _ex is None:
                                _txt = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                                _ex = any(c.isdigit() for c in _txt)
                                self._rep_exempt_tokens[tok_id] = _ex
                            if _ex:
                                continue

                        if logits[0, tok_id] > 0:
                            logits[0, tok_id] /= _pen_val
                        else:
                            logits[0, tok_id] *= _pen_val

            # Apply Factual Logit Bias
            srl_state = getattr(self.manager, "_session_srl", {}).get(session_id)
            if srl_state is not None:
                # ── Helper token set (needed for both VSL masking and penalty below) ──
                from native_core.srl.factual_alignment import get_helper_token_ids
                helper_ids = get_helper_token_ids(self.tokenizer)

                # +7.0 factual token bias — raised from +3 to overcome LM paraphrase prior.
                # At +3 the model could still prefer "converge" over "coalesce" if its
                # prior favoured the former by more than 3 logit units. At +7 the gap
                # is wide enough that source-exact tokens reliably win.
                if getattr(srl_state, "current_step_factual_tokens", None):
                    current_entity = getattr(srl_state, "current_entity_id", -1)
                    entity_ids = getattr(srl_state, "current_step_sequence_entity_ids", [])
                    is_prime_list = getattr(srl_state, "current_step_sequence_is_prime", [])
                    
                    if current_entity != -1:
                        entity_factual_tokens = set()
                        for i, seq in enumerate(srl_state.current_step_factual_sequences):
                            seq_eid = entity_ids[i] if i < len(entity_ids) else -1
                            seq_is_prime = is_prime_list[i] if i < len(is_prime_list) else False
                            if seq_eid == -1 or seq_eid == current_entity or seq_is_prime:
                                entity_factual_tokens.update(seq)
                        for tok_id in entity_factual_tokens:
                            if tok_id < logits.shape[-1]:
                                logits[0, tok_id] += 7.0
                    else:
                        for tok_id in srl_state.current_step_factual_tokens:
                            if tok_id < logits.shape[-1]:
                                logits[0, tok_id] += 7.0

                # +7.0 VSL active-candidate boost
                active_candidates = getattr(srl_state, "vsl_active_candidates", [])
                if active_candidates:
                    for suffix in active_candidates:
                        if suffix and suffix[0] < logits.shape[-1]:
                            logits[0, suffix[0]] += 7.0

                # -3.5 anti-hallucination penalty ─────────────────────────────────
                # Threshold lowered 0.55→0.4 (matches SFA activation) and magnitude
                # raised -2.5→-3.5. Previously the condition also required active VSL
                # candidates, which was too restrictive: the penalty now fires as soon
                # as any factual match is present and similarity is sufficient.
                # This blocks "more complex", "distinct structures", spurious equations.
                if (getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.4
                        and not getattr(srl_state, "dual_entity_mode", False)
                        and getattr(srl_state, "current_step_factual_tokens", None)):
                    factual_set = srl_state.current_step_factual_tokens
                    _vocab = logits.shape[-1]
                    _excl = [t for t in list(factual_set) + list(helper_ids) if 0 <= t < _vocab]
                    if _excl:
                        _excl_t = torch.tensor(_excl, dtype=torch.long, device=logits.device)
                        _penalty_mask = torch.ones(_vocab, dtype=torch.bool, device=logits.device)
                        _penalty_mask.scatter_(0, _excl_t, False)
                        logits[0, _penalty_mask] -= 3.5

                # +10.0 transition bias — raised from +4 to strongly enforce known
                # token sequences. Fixes inverted/collapsed relationships by making
                # the correct next token in a source sequence decisively more likely.
                last_token = generated[-1] if generated else None
                if last_token is not None and getattr(srl_state, "current_step_factual_sequences", None):
                    transition_candidates = set()
                    current_entity = getattr(srl_state, "current_entity_id", -1)
                    entity_ids = getattr(srl_state, "current_step_sequence_entity_ids", [])
                    for i, seq in enumerate(srl_state.current_step_factual_sequences):
                        seq_entity = entity_ids[i] if i < len(entity_ids) else -1
                        if current_entity != -1 and seq_entity != -1 and seq_entity != current_entity:
                            continue  # skip cross-entity transitions
                        for idx, tok in enumerate(seq[:-1]):
                            if tok == last_token:
                                transition_candidates.add(seq[idx + 1])
                    for tok_id in transition_candidates:
                        if tok_id < logits.shape[-1]:
                            logits[0, tok_id] += 10.0

            # Apply Dynamic Temperature Scaling (Option 1)
            effective_temperature = temperature
            if srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55:
                max_sim = srl_state.current_step_max_similarity
                effective_temperature = temperature * (1.0 - max_sim * 0.95)

            # SFA threshold aligned to 0.55: at 0.3 almost every topical entry matches,
            # activating the VSL and forcing generation from a mixed-category token set.
            # At 0.55 only high-confidence, specific retrieval triggers the constraint.
            sfa_active = (
                srl_state is not None
                and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55
                and bool(getattr(srl_state, "current_step_factual_sequences", None))
            )

            # LM-VSL (Logit Masking) — guard against empty sequences; without this
            # get_allowed_tokens_vsl returns only helper words, locking generation.
            if sfa_active:
                from native_core.srl.factual_alignment import (
                    get_allowed_tokens_vsl, get_structural_helper_token_ids)
                structural_helper_ids = get_structural_helper_token_ids(self.tokenizer)
                allowed_ids = get_allowed_tokens_vsl(
                    srl_state, helper_ids,
                    structural_helper_ids=structural_helper_ids,
                    sfa_active=True,
                )
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[list(allowed_ids)] = False
                factual_toks = getattr(srl_state, "current_step_factual_tokens", None)
                if factual_toks:
                    valid_factual_toks = [t for t in factual_toks if 0 <= t < logits.shape[-1]]
                    mask[valid_factual_toks] = False
                
                max_sim = getattr(srl_state, "current_step_max_similarity", 0.0)
                if max_sim >= 0.70:
                    logits[0, mask] = -65000.0   # hard: verbatim extraction mode
                else:
                    logits[0, mask] -= 7.0        # soft: guided but escapable

            # Sample
            next_id = _compiled_sample_fn(logits, effective_temperature, top_p)
            next_id_val = next_id.item()

            # Strict Factual Alignment (SFA) State Update and Loop Check
            if srl_state is not None:
                srl_state.recent_generated_tokens.append(next_id_val)
                srl_state.generated_token_slots.append(srl_state.ordered_slot_ids[-1] if srl_state.ordered_slot_ids else 0)
                srl_state.update_query_segment(next_id_val)
                srl_state.update_dynamic_anchors(self.stop_token_ids)

            if srl_state is not None:
                from native_core.srl.factual_alignment import update_vsl_state, get_helper_token_ids
                helper_ids = get_helper_token_ids(self.tokenizer)
                update_vsl_state(next_id_val, srl_state, helper_ids)
                
                if sfa_active and getattr(srl_state, "vsl_consecutive_helpers", 0) >= 16:
                    uncertainty_suffix = " [uncertain: details missing in source]"
                    uncertainty_tokens = self.tokenizer.encode(uncertainty_suffix, add_special_tokens=False)
                    for t_id in uncertainty_tokens:
                        generated.append(t_id)
                        if hasattr(self.manager, "register_prefill_tokens"):
                            self.manager.register_prefill_tokens(session_id, torch.tensor([t_id], dtype=torch.long))
                    break

            generated.append(next_id_val)
            if hasattr(self.manager, "register_prefill_tokens"):
                self.manager.register_prefill_tokens(session_id, torch.tensor([next_id_val], dtype=torch.long))

            if srl_state is not None and hasattr(srl_state, "save_step_state"):
                srl_state.save_step_state(len(generated))

            # Predictive Prefetching
            if os.environ.get("DKV_PREDICTIVE_PAGING", "0") == "1" and srl_state is not None:
                prefetch_slots = set()

                # 1. Lexical: Inverted index lookup on the new generated token
                if srl_state.inverted_index is not None and next_id_val in srl_state.inverted_index.occurrences:
                    for slot, _, _ in srl_state.inverted_index.occurrences[next_id_val]:
                        prefetch_slots.add(slot)

                # 2. Graph neighbors of currently active slots (current_step_slots)
                active_slots = getattr(srl_state, "current_step_slots", None)
                if active_slots is not None:
                    active_slots_set = set(active_slots.cpu().tolist())
                    expanded_slots = srl_state.expand_neighborhood(active_slots_set)
                    prefetch_slots.update(expanded_slots - active_slots_set)

                    # 3. Next chronological blocks (slot + 1)
                    for slot in active_slots_set:
                        prefetch_slots.add(slot + 1)

                # Issue prefetches across all layers
                all_slots = set(srl_state.ordered_slot_ids)
                for slot in prefetch_slots:
                    if slot in all_slots:
                        for l_idx in range(self.manager.num_layers):
                            blocks = self.manager.session_blocks.get(session_id, {}).get(l_idx, [])
                            for idx, b in enumerate(blocks):
                                if b.pool_idx == slot:
                                    self.manager.prefetch(session_id, l_idx, idx)
                                    break

            # Factual Early Stopping (Option 2 Extension)
            stop_generation = False
            if max_new_tokens < 64 and srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.5:
                if getattr(srl_state, "current_step_factual_sequences", None):
                    for seq in srl_state.current_step_factual_sequences:
                        if len(seq) >= 5 and len(generated) >= len(seq):
                            if generated[-len(seq):] == list(seq):
                                stop_generation = True
                                break
            if stop_generation:
                break

            # next_id_val already holds next_id.item() from the sample step
            # above. Calling .item() again is a SECOND device sync on the same
            # value, once per generated token -- torch.cuda.set_sync_debug_mode
            # named this line and line 1474 as the only per-token syncs in the
            # generate loop.
            if next_id_val in self.stop_token_ids:
                break

            # Pass the correct absolute position so RoPE rotates at the right angle.
            # past_key_values is always None (DKV manages KV internally), so without
            # this the model would wrongly use position 0 for every decode token.
            pos_tensor = pos_cache[cur_pos].view(1, 1)
            input_ids = next_id.unsqueeze(0)

            # Finalize any completed CPU background compressions
            if hasattr(self.manager, "finalize_compressed_blocks"):
                self.manager.finalize_compressed_blocks()

            # ── Decode forward: choose execution path ──────────────────────
            is_mps = (self.device == "mps" or
                      (isinstance(self.device, torch.device) and self.device.type == "mps"))
            is_cuda = torch.cuda.is_available() and not is_mps

            _time_token_start = None
            _time_attn_flag = os.environ.get("DKV_TIME_ATTN") == "1"
            if _time_attn_flag:
                import time as _tw
                if is_mps:
                    try: torch.mps.synchronize()
                    except Exception: pass
                _time_token_start = _tw.perf_counter()

            if is_mps and hasattr(torch, "mps") and hasattr(torch.mps, "capture_to_graph"):
                # MPS path: Metal graph capture eliminates driver overhead
                with torch.mps.capture_to_graph():
                    outputs = self.model(
                        input_ids=input_ids,
                        position_ids=pos_tensor,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
            elif is_cuda and _HAS_CUDA_GRAPH_RUNNER:
                # CUDA path: CUDA Graph runner. The runner captures on first call
                # (after 3 warmup passes) then replays the static graph for all
                # subsequent decode steps — eliminates kernel launch overhead.
                # The Triton kernel uses pool indices that are static tensors, so
                # graph capture is safe as long as the block layout doesn't change.
                # We invalidate the graph whenever a new prefill runs (new session).
                if not hasattr(self, "_cuda_graph_runner") or self._cuda_graph_runner is None:
                    self._cuda_graph_runner = CUDAGraphDecodeRunner() if _HAS_CUDA_GRAPH_RUNNER else None

                if self._cuda_graph_runner is not None:
                    # Decide mutation-out for THIS step before anything runs a
                    # forward. capture() runs the forward too, so publishing it
                    # later would let the captured graph and the replayed steps
                    # disagree about whether the forward mutates state.
                    self._dkv_publish_mutation_out(session_id)
                    if (getattr(_dkv_attn_mod, "_MUTATION_OUT_ACTIVE", False)
                            and not getattr(self, "_dkv_primed", False)):
                        self._dkv_primed = True
                        try:
                            self._dkv_apply_pending_mutation(session_id)
                        except Exception:                        # noqa: BLE001
                            pass
                    if (not self._cuda_graph_runner.is_captured()
                            and not getattr(self, "_dkv_capture_giveup", False)):
                        try:
                            # Hand capture() the cache the DKV bypass actually
                            # mutates so it can roll back its own warmup writes.
                            # Without it the runner declines to capture rather
                            # than leave the KV write index ahead of position_ids.
                            _dkv_cache = None
                            try:
                                _ws = getattr(self.manager, "decode_workspace", None)
                                _sids = getattr(self.model, "_dkv_session_ids", None)
                                _sid = (_sids[0] if _sids else None)
                                if _ws is not None and _sid in _ws:
                                    _dkv_cache = _ws[_sid].get("dense_cache")
                            except Exception:
                                _dkv_cache = None

                            # Advertise the static-state ABI only for the case
                            # that has actually been shown replayable: the
                            # BYPASS path under DKV_GRAPH_SAFE_DECODE, where the
                            # only decode state is a StaticCache whose write
                            # index is a device tensor updated in place.
                            #
                            # A dense_cache exists only while the session is
                            # bypassed, so its presence IS the bypass test. Once
                            # DKV engages, decode runs through the routed sparse
                            # path, whose block routing is recomputed in Python
                            # every step and would be frozen by replay — that
                            # path still needs its own static ABI. The runner
                            # applies one further refusal of its own for hybrid
                            # models with recurrent layers.
                            # Two capturable configurations now:
                            #   bypass  -- short context, state is the dense
                            #              StaticCache, gated on the cache existing
                            #   routed  -- DKV engaged, state is the pool + dense
                            #              window; capturable only once routing is
                            #              fixed-shape (DKV_GRAPH_SAFE_ROUTING),
                            #              because torch.nonzero otherwise makes
                            #              every downstream shape data-dependent
                            # Only the BYPASS path is capturable today.
                            #
                            # The routed path now CAPTURES successfully once
                            # routing is fixed-shape (DKV_GRAPH_SAFE_ROUTING
                            # removes the two torch.nonzero compactions), but it
                            # does not REPLAY correctly: the dense window of
                            # recently generated tokens is rebuilt in Python every
                            # step by assemble_dense_window_kv, and replay runs no
                            # Python, so the graph keeps attending the recent
                            # context frozen at capture time and the text diverges
                            # from eager.
                            #
                            # CORRECTED 2026-08-16. This block used to say routed
                            # replay "is also not faster -- 85 ms/token against
                            # eager's 79", and concluded the host time was mostly
                            # OUTSIDE the captured region. Both halves were wrong,
                            # and the reason is instructive.
                            #
                            # The capture never succeeded. It raises
                            # cudaErrorStreamCaptureInvalidated, and the caller
                            # swallowed it with a bare `except Exception: pass`,
                            # so the run silently fell back to eager. The "85 vs
                            # 79" was eager-with-graph-overhead against eager --
                            # replay was never measured. The except now logs once.
                            #
                            # The host time is INSIDE the forward, not outside:
                            # 89.6% of decode wall is inside model.forward (41.94
                            # of 46.81 ms/token on Qwen3.5-2B at 16k), and against
                            # 27.19 ms/token of CUDA kernels that leaves ~14.7
                            # ms/token of host INSIDE the capturable region
                            # against 4.9 outside. So a working graph has
                            # something real to win -- roughly 30% of decode.
                            #
                            # WHAT BLOCKED CAPTURE -- now fixed. A first pass
                            # blamed _sparse_prefill_filter_blocks, which was an
                            # ATTRIBUTION ERROR: generate() re-prefills on every
                            # call and prefill runs through the same
                            # model.forward, so prefill's syncs were being counted
                            # as decode's. Filtering to q_len == 1 gave the real
                            # list, and it was much smaller:
                            #   get_cached_decode_blocks x2  cpu_indices.to(device)
                            #                                and cpu_anchors.to()
                            #                                -- pageable H2D copies
                            #   dkv_forward:2530             torch.equal on CUDA
                            #                                tensors (host bool)
                            #   dkv_forward:1973             .cpu() on a key trail
                            # The H2D copies now stage through pinned memory
                            # (kv_runtime_manager.pinned_to_device); the other two
                            # were already gated by DKV_GRAPH_SAFE_DECODE.
                            # Inside-forward syncs: 62 -> 6 -> 0.
                            #
                            # CAPTURE NOW SUCCEEDS on the routed path, and replay
                            # IS faster: wall-clock decode 16.59 -> 23.38 tok/s at
                            # 16k on Qwen2.5-1.5B, a 1.41x speedup. (Do not
                            # measure this with DKV_TIME_ATTN -- under replay the
                            # DKV Python never runs, so it emits no per-token
                            # lines and reports a fictional 1449 tok/s. Use
                            # colab/decode_wall_vs_timer.py.)
                            #
                            # STILL GATED OFF, and the reason is bigger than the
                            # dense window. Making that one write index device-
                            # resident is NOT sufficient, which is worth stating
                            # because it looks like it should be.
                            #
                            # dkv_forward calls kv_manager.ingest_streaming() on
                            # every decode token, INSIDE the forward. That call
                            # appends the new K/V to the tail block, advances
                            # _active_fill, finalises a block when it fills, and
                            # triggers compression. All of it is Python. Graph
                            # replay runs NO Python, so under replay the token is
                            # never ingested at all: _active_fill never advances,
                            # blocks never finalise, compression never fires, and
                            # the KV store freezes wholesale. The frozen dense
                            # window is the visible symptom of that, not the cause.
                            #
                            # So routed replay needs the block LIFECYCLE to be
                            # device-resident -- append, finalise, and compression
                            # triggering -- not just one index. That is the
                            # "static-state ABI" this file originally called for,
                            # and it is a redesign.
                            #
                            # DEFERRED INGEST WAS EVALUATED AND DOES NOT WORK.
                            # The idea: replay for token t, then ingest from
                            # Python outside the captured region. Two findings
                            # killed it, both measured rather than argued.
                            #
                            # 1. Ingest runs BEFORE the attention dispatch, so the
                            #    dense window supplies the SELF-attention term.
                            #    Deferring drops it silently -- quality decays with
                            #    nothing raising. Fixable, by giving
                            #    attend_with_remat an explicit curr_k/curr_v row.
                            #
                            # 2. ROUTING is the real blocker, and arithmetic ends
                            #    it. Routing is recomputed in Python from block
                            #    lists, so a replayed graph freezes it and the
                            #    graph must be re-captured whenever it changes.
                            #    Capture costs 288 ms (measured). Replay saves
                            #    ~14.7 ms/token of host. Break-even is ~20 tokens
                            #    of replay per capture -- but DKV_REMAT_INTERVAL
                            #    freezes routing for only 4. Re-capturing on every
                            #    routing change is a 5x NET LOSS, and holding
                            #    routing for 20 tokens is five times the staleness
                            #    that interval was deliberately set to avoid.
                            #
                            # So the routed 1.41x requires routing itself to be
                            # DEVICE-RESIDENT -- block selection computed inside
                            # the graph from device-resident block metadata, so no
                            # re-capture is ever needed. That is the original
                            # "static-state ABI" call, now with numbers behind it.
                            # DKV_GRAPH_FORCE_ROUTED=1 is a MEASUREMENT-ONLY
                            # override: it lets the routed path capture and
                            # replay so the SPEED CEILING of a correct graph can
                            # be measured, and it PRODUCES WRONG TEXT because the
                            # dense window stays frozen at capture time. It exists
                            # because the "not faster" note above is from an older
                            # build and deciding whether to fund the device-
                            # resident rewrite needs a current number, not a
                            # stale one. Never set it in a serving path.
                            # OPT-IN, and the measurement that decided it.
                            # Enabling routed capture by default was tried and
                            # REVERTED: it requires DKV_GRAPH_MUTATION_OUT, which
                            # moves per-layer ingest and window assembly out of
                            # the forward into a Python loop in the wrapper. That
                            # loop costs one iteration PER ATTENDED LAYER, so its
                            # price scales with layer count while the graph payoff
                            # does not:
                            #
                            #   Qwen3.5-2B  (6 attended layers)  32k: 15.80 vs
                            #       15.57 and 15.12 vs 14.79 tok/s -- slightly
                            #       FASTER with mutation-out
                            #   Qwen2.5-1.5B (28 attended layers) 32k: 11.74 vs
                            #       12.94 and 12.07 vs 13.54 tok/s -- about 9%
                            #       SLOWER, in both interleaved rounds
                            #
                            # At 32k the selectivity gate declines the graph, so
                            # that cost buys nothing there. Enabling it globally
                            # therefore speeds up 16k (17.3 -> 10.2 s wall,
                            # byte-identical) while regressing long context on
                            # wide models, which is not a trade to make silently.
                            #
                            # Turn BOTH on together for the win where routing is
                            # non-selective: DKV_GRAPH_MUTATION_OUT=1 and
                            # DKV_GRAPH_FORCE_ROUTED=1.
                            # DKV_FAST_DECODE=1 turns this on together with
                            # mutation-out; either alone is useless (see the
                            # note on _FAST_DECODE in dkv_attention).
                            _fast_dec = os.environ.get("DKV_FAST_DECODE", "0") == "1"
                            _force_routed = os.environ.get(
                                "DKV_GRAPH_FORCE_ROUTED",
                                "1" if _fast_dec else "0") == "1"
                            # ROUTED CAPTURE IS EXACT ONLY WHERE ROUTING IS
                            # NON-SELECTIVE, and that is checkable rather than
                            # hoped for. Replay cannot re-run the Python router,
                            # so the routed set it captured is frozen for the
                            # whole replay. When the router would have selected
                            # every block anyway -- n_blocks <= K -- freezing it
                            # is a NO-OP and replay is exact by construction.
                            #
                            # Measured both sides of that line on Qwen2.5-1.5B:
                            #   16k, 15 blocks, K=16 (non-selective)
                            #       byte-identical at 48/64/96 tokens, and
                            #       15.7 -> 10.6 s wall at 96 (1.48x)
                            #   32k, 31 blocks, K=16 (selective)
                            #       drifts under FORCED capture, coherent text,
                            #       i.e. staleness not corruption
                            #
                            # The md5 pair originally recorded here (7c291f42
                            # against "eager 9a9cbc07") did NOT show that. Both
                            # arms ran without DKV_DETERMINISTIC, and at 32k the
                            # decode attention's own reduction changes the answer
                            # between two runs of the SAME config -- so that pair
                            # measured the nondeterminism, not the drift. Rerun
                            # with DKV_DETERMINISTIC=1, eager and the gated
                            # decline both give 7c291f42ece7d897.
                            #
                            # Refreshing it does not rescue the selective case.
                            # Eager alternation CORRUPTS (interval 2/4/8 breaks
                            # after ~1/~8/~14 tokens) and frequent re-capture is
                            # both inexact at length and slower than eager (288 ms
                            # per capture against 14.7 ms/token saved: at
                            # recapture=2, 25.5 s vs eager's 17.9 s and a
                            # different md5 by 64 tokens). So the gate is the
                            # honest ship, not a placeholder for a refresh.
                            # ONE implementation of the gate, shared with
                            # _dkv_publish_mutation_out. It was inlined here when
                            # capture was its only consumer; mutation-out now asks
                            # the same question every step, and two copies of a
                            # correctness gate drifting apart is not a risk worth
                            # taking for a dozen lines.
                            _routing_selective = False
                            if _force_routed:
                                _routing_selective = self._dkv_routing_selective(
                                    session_id)
                                if _routing_selective and not getattr(
                                        self, "_graph_sel_logged", False):
                                    self._graph_sel_logged = True
                                    print(f"[DKV] routed CUDA graph declined: "
                                          f"compressed blocks exceed K, so routing "
                                          f"is selective and a frozen routed set "
                                          f"would drift. "
                                          f"DKV_GRAPH_ALLOW_SELECTIVE=1 to override.",
                                          file=sys.stderr, flush=True)
                            if os.environ.get("DKV_GRAPH_ALLOW_SELECTIVE") == "1":
                                _routing_selective = False
                            self.model._dkv_cuda_graph_safe = bool(
                                (_force_routed and not _routing_selective) or (
                                    _dkv_cache is not None
                                    and os.environ.get("DKV_GRAPH_SAFE_DECODE", "0") == "1"))
                            self._cuda_graph_runner.capture(
                                self.model, input_ids, pos_tensor, cache=_dkv_cache)
                            # SNAPSHOT the ingest references produced DURING
                            # capture. This is the whole trick, and getting it
                            # wrong is why replay produced degenerate text:
                            # _pending_ingest is rewritten by every EAGER step
                            # with ordinary torch tensors, which replay never
                            # touches, so draining it after a replay re-ingested
                            # the last eager token again and again. The refs
                            # recorded during capture point into the GRAPH's own
                            # memory pool, which replay does rewrite -- those are
                            # the ones that carry fresh values.
                            _pi = self.manager.__dict__.get("_pending_ingest")
                            if _pi:
                                self.manager.__dict__["_graph_ingest_refs"] = dict(_pi)
                            # Record WHAT WAS ROUTABLE AT CAPTURE. The routed set
                            # is pinned into fixed-address buffers, so a replay
                            # attends whatever slots those buffers held when the
                            # graph was recorded. That is exact while the block
                            # set is unchanged -- and the selectivity gate makes
                            # sure it starts that way -- but blocks keep being
                            # COMPRESSED during decode, so a session that
                            # captured at 15 blocks is attending 15 of 16 a few
                            # tokens later, with the newest content the one thing
                            # missing. See the invalidation below.
                            self._graph_ncomp = self._dkv_block_counts(
                                session_id)[0]
                            # Snapshot WHAT THE FORWARD BOUND DURING CAPTURE.
                            # Comparing against the live _fwd_ptrs only compares
                            # the last EAGER forward, which says nothing about the
                            # graph -- this is the pair that actually matters.
                            _fp = self.manager.__dict__.get("_fwd_ptrs")
                            if _fp:
                                self.manager.__dict__["_graph_fwd_ptrs"] = {
                                    k: dict(v) for k, v in _fp.items()}
                        except Exception as _cap_err:
                            # Log ONCE. A bare `pass` here makes a failed capture
                            # indistinguishable from eager decode, so a benchmark
                            # can report "graph replay" numbers while no graph
                            # exists -- which is exactly what happened when the
                            # routed-path speed ceiling was first measured.
                            if not getattr(self, "_graph_capture_err_logged", False):
                                self._graph_capture_err_logged = True
                                print(f"[DKV] CUDA graph capture FAILED "
                                      f"({type(_cap_err).__name__}: {_cap_err}); "
                                      f"continuing in eager decode.",
                                      file=sys.stderr, flush=True)
                        # ONE attempt decides it for the session. capture() is
                        # retried on every step while is_captured() is False, so
                        # without this a session that can never capture pays the
                        # attempt AND keeps deferring mutation for a graph that
                        # is never coming. Whatever refused it -- selectivity, a
                        # missing decode cache, an exception -- the answer will
                        # not change mid-session.
                        if not self._cuda_graph_runner.is_captured():
                            self._dkv_capture_giveup = True
                            self._dkv_publish_mutation_out(session_id)


                    # ROUTING REFRESH. Replay executes no Python, so the routed
                    # set pinned in _stabilise_routed_set's buffers would never
                    # advance. Running ONE EAGER step every N tokens lets the real
                    # forward re-route and rewrite those buffers in place, which
                    # every subsequent replay then reads -- no re-capture, and no
                    # second router implementation to drift from the first.
                    #
                    # N is the remat interval, so routing staleness under replay is
                    # exactly the staleness DKV_REMAT_INTERVAL already ships and
                    # accepts (the remat cache freezes the routed set for the same
                    # window). Not a new trade -- the same one, made explicit.
                    if not hasattr(self, "_dkv_step_idx"):
                        self._dkv_step_idx = 0
                    _mo = getattr(_dkv_attn_mod, "_MUTATION_OUT_ACTIVE", False)
                    # NEVER alternate eager and replay. Running one eager forward
                    # between replays was the original refresh design and it is
                    # what corrupted the state -- proven by running the cases in
                    # order: with NO eager step at all, replay reproduces eager
                    # byte for byte, and every interval that introduces eager
                    # steps degrades in proportion to how many it introduces.
                    #
                    # Routing is instead refreshed by RE-CAPTURING, which reruns
                    # the real forward and rebuilds every pinned buffer from
                    # scratch. Capture costs 288 ms and replay saves ~14.7
                    # ms/token, so re-capturing every DKV_GRAPH_RECAPTURE tokens
                    # costs 288/N ms/token against 14.7 saved -- net positive for
                    # any N above ~20, and 64 leaves routing staleness bounded at
                    # 64 tokens.
                    _force_eager = False
                    if _mo and self._cuda_graph_runner.is_captured():
                        try:
                            _recap = int(os.environ.get("DKV_GRAPH_RECAPTURE", "64"))
                        except ValueError:
                            _recap = 64
                        if _recap > 0 and self._dkv_step_idx > 0                                 and (self._dkv_step_idx % _recap) == 0:
                            self._cuda_graph_runner.invalidate()
                        # INVALIDATE WHEN THE BLOCK SET CHANGES. This is the
                        # frozen-routing fix, and it is narrower than re-routing.
                        #
                        # The routed set is pinned to fixed addresses, so replay
                        # attends the slots recorded at capture. The selectivity
                        # gate guarantees that is EXACT at capture time -- routing
                        # is non-selective, so "the routed set" is "every block".
                        # What it cannot guarantee is that it stays true: blocks
                        # keep getting compressed as decode proceeds, and the
                        # moment block 16 appears, a graph captured over 15 is
                        # attending everything EXCEPT the newest content. That is
                        # a silent quality loss, not a crash.
                        #
                        # IT IS NOT, HOWEVER, WHY --fastdc DIVERGES FROM EAGER.
                        # That was the hypothesis this was written to test and it
                        # failed. Measured on Qwen2.5-1.5B with
                        # DKV_DETERMINISTIC=1, eager against replayed: 4k and 16k
                        # are byte-identical to their pre-fix md5s and this
                        # invalidation never even fires there, while 8k fires
                        # (6 -> 7 blocks), changes its md5, and still does not
                        # match eager. So the divergence has a second cause that
                        # is NOT the routed set going stale, and it is still
                        # open. Kept anyway: attending 15 of 16 blocks with the
                        # newest content missing is wrong on its own terms,
                        # whatever else is also wrong.
                        #
                        # A count comparison is enough and costs 4.6 us: blocks
                        # only ever move INTO the compressed set during decode.
                        # Re-capture then rebuilds every pinned buffer from a
                        # real forward, which is the refresh mechanism this
                        # design already documents -- and unlike the eager
                        # alternation that was tried and retracted, no step ever
                        # runs against half-updated state.
                        _now_ncomp = self._dkv_block_counts(session_id)[0]
                        if (_now_ncomp is not None
                                and getattr(self, "_graph_ncomp", None) is not None
                                and _now_ncomp != self._graph_ncomp):
                            self._cuda_graph_runner.invalidate()
                            if not getattr(self, "_graph_ncomp_logged", False):
                                self._graph_ncomp_logged = True
                                print(f"[DKV] routed graph invalidated: compressed "
                                      f"blocks {self._graph_ncomp} -> {_now_ncomp}, "
                                      f"so the pinned routed set no longer covers "
                                      f"the pool. Re-capturing.",
                                      file=sys.stderr, flush=True)
                    self._dkv_step_idx += 1
                    self._dkv_last_was_replay = (
                        self._cuda_graph_runner.is_captured() and not _force_eager)
                    if self._cuda_graph_runner.is_captured() and not _force_eager:
                        try:
                            outputs = self._cuda_graph_runner.run(input_ids, pos_tensor)
                        except Exception:
                            outputs = self.model(
                                input_ids=input_ids,
                                position_ids=pos_tensor,
                                past_key_values=past_kv,
                                use_cache=True,
                            )
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            position_ids=pos_tensor,
                            past_key_values=past_kv,
                            use_cache=True,
                        )
                    self._dkv_apply_pending_mutation(session_id)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        position_ids=pos_tensor,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
            else:
                # CPU / fallback eager
                outputs = self.model(
                    input_ids=input_ids,
                    position_ids=pos_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            if _time_attn_flag and _time_token_start is not None:
                # SYNCHRONISE BEFORE READING THE CLOCK, on CUDA as well as MPS.
                # Without it this is a HOST timer, and the two things worth
                # measuring here differ mostly in host cost -- so it flatters
                # exactly what it is used to judge. A CUDA graph replay is one
                # launch: the host returns while the GPU is still working, and
                # this reported 4.63 ms/token (215 tok/s) for a path measured at
                # ~11-13 tok/s end to end, an 87% "win" that was entirely the
                # missing sync. Eager decode happened to read about right only
                # because that path is host-bound.
                if self.device == "mps":
                    try: torch.mps.synchronize()
                    except Exception: pass
                elif self.device == "cuda":
                    try: torch.cuda.synchronize()
                    except Exception: pass
                _token_ms = (_tw.perf_counter() - _time_token_start) * 1000
                print(f"[DKV_TIME_ATTN] total_token={_token_ms:.2f}ms", flush=True)

            logits = outputs.logits[:, -1, :]
            # DKV_LOGIT_TRACE=1 -- per-step logit fingerprint, for finding the
            # FIRST step at which a replayed decode parts from an eager one.
            #
            # Every INPUT to the replayed attention has been accounted for
            # ([DUMP] for the wrapper-owned state, [POOL] for the compressed
            # pool) and none of them move, so the remaining question is where the
            # OUTPUT first differs. Diff two runs line by line: the step where
            # argmax changes is where the text splits, but sum/max move first and
            # by how much says whether it is drift or a wrong read.
            if os.environ.get("DKV_LOGIT_TRACE") == "1":
                _lf = logits.float()
                print(f"[LOGIT] step={getattr(self, '_dkv_step_idx', -1):3d} "
                      f"replay={int(bool(getattr(self, '_dkv_last_was_replay', False)))} "
                      f"argmax={int(_lf.argmax())} "
                      f"max={float(_lf.max()):.6f} "
                      f"sum={float(_lf.sum()):.4f}",
                      file=sys.stderr, flush=True)
            past_kv = outputs.past_key_values
            cur_pos += 1

            # Advance the CAD prior stream by the SAME token (plain eager forward,
            # device-agnostic), refreshing its logits for the next combine. Stop
            # once the step cap is hit and decode the rest at full tps.
            if _cad_on:
                _cad_step += 1
                if _cad_max_steps > 0 and _cad_step >= _cad_max_steps:
                    _cad_on = False
                    if _cad_sid is not None:
                        try:
                            self.manager.clear_session(_cad_sid)
                        except Exception:
                            pass
                        _cad_sid = None
                else:
                    try:
                        self.model._dkv_session_ids = [_cad_sid]
                        _pp = torch.tensor([[_cad_pos]], dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            _po = self.model(input_ids=input_ids, position_ids=_pp,
                                             past_key_values=_cad_past, use_cache=True)
                        _cad_prior_logits = _po.logits[:, -1, :]
                        _cad_past = _po.past_key_values
                        _cad_pos += 1
                    finally:
                        self.model._dkv_session_ids = [session_id]

        # Release the CAD prior stream session.
        if _cad_sid is not None:
            try:
                self.manager.clear_session(_cad_sid)
            except Exception:
                pass

        # Store the generated tokens to the session token cache
        self._session_token_ids[session_id] = generated

        # Clear loop detection state for this session after generation completes
        self._hf_loop_detected = False
        self._hf_loop_idx = None

        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return _normalize_references(decoded)

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def rollback_session(self, session_id: str, target_len: int, clear_srl: bool = False):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.rollback_session(session_id, target_len, clear_srl=clear_srl)

    def clone_session(self, src_sid: str, dst_sid: str):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.clone_session(src_sid, dst_sid)

    def clear_session(self, session_id: str):
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.clear_session(session_id)

    def _custom_sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def close(self):
        """Explicitly release all resources to prevent memory leaks and circular reference accumulation."""
        if hasattr(self, "manager") and self.manager is not None:
            try:
                self.manager.close()
            except Exception as e:
                print(f"[DKV] Warning during manager close: {e}")
            self.manager = None
        
        self.model = None
        self.tokenizer = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def __del__(self):
        self.close()

import sys
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

if sys.platform == "darwin" and _HAS_MLX and os.environ.get("DKV_FORCE_PYTORCH") != "1":
    try:
        from serving.mlx_dkv_wrapper import MLXDKVWrapper as DKVHFWrapper
        print("[DKV] macOS + MLX detected: using native MLX DKV wrapper.")
    except Exception as e:
        print(f"[DKV] Warning: Failed to import MLX wrapper ({e}), falling back to PyTorch.")
        DKVHFWrapper = PyTorchDKVHFWrapper
else:
    DKVHFWrapper = PyTorchDKVHFWrapper
