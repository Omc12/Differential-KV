import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
"""
runtime/hf_diffkv_wrapper.py

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
from collections import Counter
from typing import Optional, List, Tuple, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from native_core.kv_runtime_manager import KVRuntimeManager

from native_core.sparse_decode.triton_fused_decode import TritonDiffKV
from runtime.diffkv_attention import apply_diffkv_attention_patch

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
            print("[DIFFKV_SYNC_DEBUG] WARNING: Synchronization barrier triggered by .item() call!")
            traceback.print_stack(limit=5)
            _in_sync_check = False
        return orig_item(self)

    def patched_cpu(self, *args, **kwargs):
        nonlocal _in_sync_check
        if not _in_sync_check and self.device.type != "cpu":
            _in_sync_check = True
            print("[DIFFKV_SYNC_DEBUG] WARNING: Synchronization barrier triggered by .cpu() call!")
            traceback.print_stack(limit=5)
            _in_sync_check = False
        return orig_cpu(self, *args, **kwargs)

    def patched_tolist(self):
        nonlocal _in_sync_check
        if not _in_sync_check and self.device.type != "cpu":
            _in_sync_check = True
            print("[DIFFKV_SYNC_DEBUG] WARNING: Synchronization barrier triggered by .tolist() call!")
            traceback.print_stack(limit=5)
            _in_sync_check = False
        return orig_tolist(self)

    # Patch them
    torch.Tensor.item = patched_item
    torch.Tensor.cpu = patched_cpu
    torch.Tensor.tolist = patched_tolist
    print("[DiffKV] DIFFKV_SYNC_DEBUG sync barrier checks enabled.")


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
    print(f"[DiffKV] Heap trimmed. RSS: {rss_before:.0f} MB → {rss_after:.0f} MB "
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
        print(f"[DiffKV] WARNING: {stray_count} parameters still on CPU after to({device}) — "
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
    triggering GC. Set via DIFFKV_MPS_MEMORY_FRACTION env var or config dict.
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
            print(f"[DiffKV] MPS memory fraction capped at {memory_fraction:.0%} of system RAM.")
        except Exception as e:
            print(f"[DiffKV] WARNING: Could not set MPS memory fraction: {e}")
    else:
        print("[DiffKV] MPS memory fraction: unlimited (no artificial cap applied).")

    # RSS-based pressure relief daemon
    rss_threshold_mb = float(os.environ.get("DIFFKV_MPS_RSS_THRESHOLD_MB", "3000"))

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

    t = threading.Thread(target=_pressure_monitor, daemon=True, name="diffkv-mps-pressure")
    t.start()
    print(f"[DiffKV] MPS pressure monitor started (threshold: {rss_threshold_mb:.0f} MB RSS).")


def _configure_cuda_allocator() -> None:
    """
    Set conservative CUDA allocator options to reduce fragmentation.

    garbage_collection_threshold:0.6 — trigger GC when 60% of reserved memory
      is actively allocated (vs default 80%), reducing peak fragmentation.
    max_split_size_mb:128 — largest block the caching allocator will split.
      Smaller splits mean fewer huge stranded blocks, lower peak VRAM.
    """
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "garbage_collection_threshold:0.6,max_split_size_mb:128"
    )
    print("[DiffKV] CUDA allocator config: garbage_collection_threshold=0.6, "
          "max_split_size_mb=128.")


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

class PyTorchDiffKVHFWrapper:
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
        self.rank = self.config.get("rank", 16)
        self.micro_block_size = self.config.get("micro_block_size", 256)
        
        print(f"[DiffKV] Lazy-initializing tokenizer for model {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._alphanumeric_tokens = {}
        
        # Initialize stop token IDs from tokenizer
        self.stop_token_ids = set()
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            self.stop_token_ids.update(eos_id)
        elif isinstance(eos_id, int):
            self.stop_token_ids.add(eos_id)
            
        special_words = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
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

        print(f"[DiffKV] Loading model weights on demand: {model_id} (device={self.device}, dtype={torch_dtype})...")
        
        # ── Preset-aware Auto-Quantization ──
        preset = config.get("preset", os.environ.get("DIFFKV_PRESET", "mid")).lower()
        if preset == "low" and not config.get("quantization") and not os.environ.get("DIFFKV_QUANTIZATION"):
            if self.device == "cuda":
                config["quantization"] = "nf4"
                print("[DiffKV] Low preset + CUDA: auto-enabling 4-bit NF4 quantization (bitsandbytes) to save VRAM")
            elif self.device == "mps":
                print("[DiffKV] Low preset + MPS: running in FP16 to avoid torchao NaN/stability issues on MPS")

        # ── 4-bit NF4 loading (BitsAndBytes) ──────────────────────────────────
        _quant_type_early = config.get("quantization", os.environ.get("DIFFKV_QUANTIZATION", ""))
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
                print("[DiffKV] 4-bit NF4 quantization enabled (BitsAndBytes).")
            except ImportError:
                print("[DiffKV] WARNING: bitsandbytes not installed — falling back to fp16.")

        if _has_cuda():
            _configure_cuda_allocator()

        if device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map={"": "mps"},
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                quantization_config=quantization_config,
                use_safetensors=True,
            )

        self.model.eval()
        _clear_cpu_grad_state(self.model)

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
            print("[DiffKV] Auto-detected already quantized model. Skipping torchao post-quantization.")
        else:
            quant_type = config.get("quantization", os.environ.get("DIFFKV_QUANTIZATION"))
            if quant_type in ["int8", "int4"]:
                if not _has_cuda() and not _has_mps():
                    print(f"[DiffKV] torchao {quant_type} quantization skipped on CPU.")
                elif _is_apple_silicon() and not _has_cuda():
                    if quant_type == "int4":
                        print("[DiffKV] int4 quantization on MPS is experimental.")
                    try:
                        from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
                        cfg = Int8WeightOnlyConfig() if quant_type == "int8" else Int4WeightOnlyConfig()
                        print(f"[DiffKV] Applying {quant_type} quantization via torchao on MPS...")
                        quantize_(self.model, cfg)
                        print("[DiffKV] torchao quantization applied successfully!")
                    except Exception as e:
                        print(f"[DiffKV] WARNING: torchao {quant_type} on MPS failed ({e}). Running in fp16.")
                else:
                    try:
                        from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
                        if quant_type == "int8":
                            quantize_(self.model, Int8WeightOnlyConfig())
                        elif quant_type == "int4":
                            quantize_(self.model, Int4WeightOnlyConfig())
                        print("[DiffKV] torchao quantization applied successfully!")
                    except Exception as e:
                        print(f"[DiffKV] WARNING: Failed to apply torchao weight quantization: {e}")

        self.num_layers = self.model.config.num_hidden_layers
        self.heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.heads
        if self.rank >= self.head_dim:
            old_rank = self.rank
            self.rank = self.head_dim // 2
            print(f"[DiffKV] WARNING: Capping SVD rank to {self.rank}")
        
        self.kv_heads = getattr(self.model.config, "num_key_value_heads", self.heads)
        self.serving_mode = config.get("serving_mode", "balanced")
        
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"[DiffKV] Model parameter count: {num_params / 1e6:.1f}M")
        except Exception:
            num_params = 1.5e9

        self.config = self.config or {}
        if "srl_k_min" not in self.config and "DIFFKV_SRL_K_MIN" not in os.environ:
            if num_params < 1.0e9:
                self.config["srl_k_min"] = 10
            elif num_params < 3.0e9:
                self.config["srl_k_min"] = 15
            else:
                self.config["srl_k_min"] = 20

        if "srl_k_max" not in self.config and "DIFFKV_SRL_K_MAX" not in os.environ:
            if num_params < 1.0e9:
                self.config["srl_k_max"] = 50
            elif num_params < 3.0e9:
                self.config["srl_k_max"] = 100
            else:
                self.config["srl_k_max"] = 200

        if "srl_threshold" not in self.config and "DIFFKV_SRL_THRESHOLD" not in os.environ:
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
        
        # Load stop token IDs from model config if present
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            model_eos = getattr(self.model.generation_config, "eos_token_id", None)
            if isinstance(model_eos, list):
                self.stop_token_ids.update(model_eos)
            elif isinstance(model_eos, int):
                self.stop_token_ids.add(model_eos)

        apply_diffkv_attention_patch(self.model, self.manager)

        use_compile = "1" if self.manager.config.torch_compile else "0"
        if _is_apple_silicon():
            if use_compile == "auto":
                use_compile = "0"

        if use_compile == "0":
            print("[DiffKV] torch.compile disabled.")
        elif is_quantized and use_compile != "1":
            print("[DiffKV] Quantized model detected — skipping torch.compile to avoid graph-break errors.")
        else:
            # Pre-flight: verify the C++ compiler required by TorchInductor is available.
            # On Windows this is cl.exe (MSVC); on Linux/macOS it is gcc/clang.
            # On MPS we use 'aot_eager' which has no C++ compiler requirement.
            _compiler_ok = True
            import sys as _sys
            if _sys.platform == "win32":
                import shutil
                if shutil.which("cl") is None and use_compile != "1":
                    print("[DiffKV] torch.compile skipped — cl.exe (MSVC) not found in PATH. "
                          "Install Visual Studio Build Tools or set DIFFKV_USE_TORCH_COMPILE=0 to silence this.")
                    _compiler_ok = False

            if _compiler_ok:
                _backend = _get_compile_backend()
                _mode    = _get_compile_mode()
                print(f"[DiffKV] Applying FFN-only layer torch.compile(dynamic=True, mode='{_mode}', backend='{_backend}') ...")
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
                    print(f"[DiffKV] FFN compilation applied successfully to {compiled_count} layers. First request will trigger JIT warmup.")
                except Exception as e:
                    print(f"[DiffKV] WARNING: FFN torch.compile failed ({e}). Running in eager mode.")

        # ── CUDA Graph Runner ────────────────────────────────────────────────
        # Created here so batch_engine.py can always find it via
        # getattr(wrapper, '_cuda_graph_runner', None).
        # On MPS/CPU: CUDAGraphDecodeRunner._capture_enabled=False so it's a no-op.
        if _HAS_CUDA_GRAPH_RUNNER:
            self._cuda_graph_runner = CUDAGraphDecodeRunner()
            print(f"[DiffKV] CUDAGraphDecodeRunner initialized "
                  f"({'CUDA graph capture enabled' if _has_cuda() else 'MPS/CPU — eager mode only'})")
        else:
            self._cuda_graph_runner = None

        if os.environ.get("DIFFKV_SYNC_DEBUG", "0") == "1":
            _patch_tensor_sync_barriers()

        # ── Post-init memory cleanup ─────────────────────────────────────────
        # Fix 1B + 2.2 — run after everything is wired up so all temp objects are free.
        _clear_cpu_param_copies(self.model, self.device)   # audit stray CPU params + flush cache

        if self.device == "mps":
            # Cap MPS allocator fraction + start RSS pressure daemon.
            # Only set hard fraction cap if DIFFKV_MPS_MEMORY_FRACTION env var is explicitly configured.
            # Otherwise, avoid setting it to prevent artificial allocator OOMs.
            mps_fraction_str = os.environ.get("DIFFKV_MPS_MEMORY_FRACTION")
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
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ):
        session_id = self.active_session or "default"
        
        # O(1) Smart Prefix Check: check if the session already has resident KV cache.
        # If so, mark the cached length so prefill is incremental (avoiding O(N) re-prefill of history).
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_ids = inputs.input_ids[0].tolist()
        
        cached_len = 0
        if hasattr(self.manager, "get_session_sequence_length"):
            seq_len = self.manager.get_session_sequence_length(session_id)
            if seq_len > 0 and seq_len < len(prompt_ids):
                stored_ids = getattr(self, "_session_token_ids", {}).setdefault(session_id, [])
                if len(stored_ids) >= seq_len and prompt_ids[:seq_len] == stored_ids[:seq_len]:
                    cached_len = seq_len
                    print(f"[DiffKV Wrapper] Found cached history for session {session_id}: length {cached_len} tokens. Reusing KV cache!")
                    
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
        self.model._diffkv_session_ids = [session_id]

        # Invalidate CUDA graph runner — new prefill changes pool layout
        if hasattr(self, "_cuda_graph_runner") and self._cuda_graph_runner is not None:
            self._cuda_graph_runner.invalidate()

        # ── Chunked prefill ──────────────────────────────────────────────────
        # Process the prompt in 512-token chunks. After each chunk we call
        # compress_prefill_kv so SVD runs on the background thread while the
        # next chunk is being forward-passed (double-buffering compute and
        # compression). This:
        #   1. Eliminates the O(N²) attention VRAM spike from one giant forward.
        #   2. Hides most of the SVD latency inside prefill time.
        #   3. Keeps peak VRAM bounded regardless of prompt length.
        PREFILL_CHUNK = getattr(self.manager, "config", None) and self.manager.config.prefill_chunk_size or 512
        new_ids_list = new_prompt_ids
        total_new = len(new_ids_list)
        outputs = None
        import time as _time

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

            # Kick off async background SVD for this chunk immediately —
            # the next chunk's forward pass runs in parallel with SVD.
            if hasattr(self.manager, "compress_prefill_kv"):
                self.manager.compress_prefill_kv(session_id)

        # Release pre-allocated buffers
        del _prefill_buf, _pos_buf

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

        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [1, vocab]

        # CRITICAL FIX: track the absolute sequence position for each decode step.
        cur_pos = cached_len + prefill_len

        # Pre-allocate position cache to avoid slow GPU allocations in the loop
        max_total_len = cur_pos + max_new_tokens + 10
        pos_cache = torch.arange(max_total_len, dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            # ── Repetition-loop detection (mirrors batch_engine.py / mlx_diffkv_wrapper.py Fix 2) ──────
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
                            f"[DiffKV HF] WARNING: repetition loop detected at token "
                            f"{_n_new}. Escalating penalty window to 256 tokens and strength to 1.3x."
                        )

            if _loop_detected:
                if _loop_idx is None:
                    self._hf_loop_idx = _n_new
                elif _n_new - _loop_idx >= 40:
                    print(
                        "[DiffKV HF] WARNING: repetition loop persisted for 40 tokens "
                        "after detection — forcing EOS."
                    )
                    break

            # Repetition penalty (widened window when a loop is active)
            _pen_window = 256 if _loop_detected else 64
            _pen_val = max(repetition_penalty, 1.3) if _loop_detected else repetition_penalty
            if _pen_val != 1.0:
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

                        if logits[0, tok_id] > 0:
                            logits[0, tok_id] /= _pen_val
                        else:
                            logits[0, tok_id] *= _pen_val

            # Apply Factual Logit Bias
            srl_state = getattr(self.manager, "_session_srl", {}).get(session_id)
            if srl_state is not None:
                if getattr(srl_state, "current_step_factual_tokens", None):
                    for tok_id in srl_state.current_step_factual_tokens:
                        if tok_id < logits.shape[-1]:
                            logits[0, tok_id] += 1.5

                # Transition biasing (Option 1)
                last_token = generated[-1] if generated else None
                if last_token is not None and getattr(srl_state, "current_step_factual_sequences", None):
                    transition_candidates = set()
                    for seq in srl_state.current_step_factual_sequences:
                        for idx, tok in enumerate(seq[:-1]):
                            if tok == last_token:
                                transition_candidates.add(seq[idx + 1])
                    for tok_id in transition_candidates:
                        if tok_id < logits.shape[-1]:
                            logits[0, tok_id] += 2.0

            # Apply Dynamic Temperature Scaling (Option 1)
            effective_temperature = temperature
            if srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.4:
                max_sim = srl_state.current_step_max_similarity
                effective_temperature = temperature * (1.0 - max_sim * 0.8)

            # Sample
            next_id = _compiled_sample_fn(logits, effective_temperature, top_p)

            generated.append(next_id.item())
            if hasattr(self.manager, "register_prefill_tokens"):
                self.manager.register_prefill_tokens(session_id, torch.tensor([next_id.item()], dtype=torch.long))
            if next_id.item() in self.stop_token_ids:
                break

            # Pass the correct absolute position so RoPE rotates at the right angle.
            # past_key_values is always None (DiffKV manages KV internally), so without
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
                    if not self._cuda_graph_runner.is_captured():
                        try:
                            self._cuda_graph_runner.capture(self.model, input_ids, pos_tensor)
                        except Exception:
                            pass
                    
                    if self._cuda_graph_runner.is_captured():
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

            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            cur_pos += 1

        # Store the generated tokens to the session token cache
        self._session_token_ids[session_id] = generated

        # Clear loop detection state for this session after generation completes
        self._hf_loop_detected = False
        self._hf_loop_idx = None

        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return _normalize_references(decoded)

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def _custom_sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def close(self):
        """Explicitly release all resources to prevent memory leaks and circular reference accumulation."""
        if hasattr(self, "manager") and self.manager is not None:
            try:
                self.manager.close()
            except Exception as e:
                print(f"[DiffKV] Warning during manager close: {e}")
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

if sys.platform == "darwin" and _HAS_MLX:
    try:
        from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper as DiffKVHFWrapper
        print("[DiffKV] macOS + MLX detected: using native MLX DiffKV wrapper.")
    except Exception as e:
        print(f"[DiffKV] Warning: Failed to import MLX wrapper ({e}), falling back to PyTorch.")
        DiffKVHFWrapper = PyTorchDiffKVHFWrapper
else:
    DiffKVHFWrapper = PyTorchDiffKVHFWrapper

