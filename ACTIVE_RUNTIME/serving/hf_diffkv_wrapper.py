import os
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

class DiffKVHFWrapper:
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
    ):
        self.model_id = model_id
        self.config = config
        # ── Device auto-detection ──────────────────────────────────────────
        self.device = device if device is not None else _get_best_device()
        print(f"[DiffKV] Device: {self.device}")
        if torch_dtype is None:
            if self.device in ("cuda", "mps"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.bfloat16
        self.mode = config.get("mode", "fp16")
        self.block_size = config.get("block_size", 256)      # S=256 → 5.2× compression
        self.rank = config.get("rank", 16)
        self.micro_block_size = config.get("micro_block_size", 256)

        
        print(f"Loading model {model_id} (device={self.device}, dtype={torch_dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._alphanumeric_tokens = {}

        # ── 4-bit NF4 loading (BitsAndBytes) ──────────────────────────────────
        # Triggered by config["quantization"] == "nf4" or DIFFKV_QUANTIZATION=nf4.
        # Reduces Qwen 2.5 1.5B from 3.1 GB → ~1.2 GB VRAM.
        # Only available on CUDA (bitsandbytes has no MPS support yet).
        _quant_type_early = config.get(
            "quantization", os.environ.get("DIFFKV_QUANTIZATION", "")
        )
        if (quantization_config is None
                and _quant_type_early == "nf4"
                and _has_cuda()):
            try:
                from transformers import BitsAndBytesConfig as _BnBConfig
                quantization_config = _BnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,   # 2nd quantization step: -15% more VRAM
                )
                torch_dtype = torch.bfloat16   # compute dtype must match
                print("[DiffKV] 4-bit NF4 quantization enabled (BitsAndBytes). "
                      "Model VRAM: ~1.2 GB vs 3.1 GB BF16.")
            except ImportError:
                print("[DiffKV] WARNING: bitsandbytes not installed — cannot use NF4 4-bit. "
                      "Install with: pip install bitsandbytes. Falling back to fp16.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            quantization_config=quantization_config
        )

        self.model.eval()
        
        # ── Auto-detect standard 4-bit / 8-bit quantization (GPTQ, AWQ, bitsandbytes) ──
        is_quantized = False
        for name, module in self.model.named_modules():
            module_class = module.__class__.__name__.lower()
            if any(q_word in module_class for q_word in ["quant", "linear4bit", "linear8bit", "wqlinear", "bnb"]):
                is_quantized = True
                break
        
        # Also check parameters datatype (quantized models might have int8 or int4 weights)
        if not is_quantized:
            for param in self.model.parameters():
                if param.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
                    is_quantized = True
                    break
        
        if is_quantized:
            print("[DiffKV] Auto-detected already quantized model (GPTQ/AWQ/Bitsandbytes). Skipping torchao post-quantization to avoid conflicts.")
        else:
            # ── Native weight-only quantization (torchao) ──
            # Skip on MPS/CPU where torchao may not support all ops yet.
            quant_type = config.get("quantization", os.environ.get("DIFFKV_QUANTIZATION"))
            if quant_type in ["int8", "int4"]:
                if not _has_cuda() and not _has_mps():
                    print(f"[DiffKV] torchao {quant_type} quantization skipped on CPU — run on GPU/MPS for best performance.")
                elif _is_apple_silicon() and not _has_cuda():
                    # MPS: int4 group quantization requires contiguous ops not yet in MPS;
                    # int8 is generally safe. Attempt it and fall back gracefully.
                    if quant_type == "int4":
                        print("[DiffKV] int4 quantization on MPS is experimental — attempting, will fall back if unsupported.")
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
                            print("[DiffKV] Applying native 8-bit weight-only quantization using torchao...")
                            quantize_(self.model, Int8WeightOnlyConfig())
                        elif quant_type == "int4":
                            print("[DiffKV] Applying native 4-bit weight-only quantization using torchao...")
                            quantize_(self.model, Int4WeightOnlyConfig())
                            
                        print("[DiffKV] torchao quantization applied successfully!")
                    except Exception as e:
                        print(f"[DiffKV] WARNING: Failed to apply torchao weight quantization: {e}")

        self.num_layers = self.model.config.num_hidden_layers
        self.heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.heads
        
        self.kv_heads = getattr(self.model.config, "num_key_value_heads", self.heads)
        self.serving_mode = config.get("serving_mode", "balanced")
        self.manager = KVRuntimeManager(
            self.num_layers,
            self.kv_heads,
            self.head_dim,
            device=device,
            rank=self.rank,
            micro_block_size=self.micro_block_size,
            serving_mode=self.serving_mode,
            tokenizer=self.tokenizer,    # ← SRL: used for stop word precomputation
            config=self.config,
        )
        self.manager.model_id = self.model_id
        self.active_session = None
        
        # Collect stop token IDs universally across all loaded models (Qwen, Llama, Mistral, etc.)
        self.stop_token_ids = set()
        
        # 1. Collect from tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            self.stop_token_ids.update(eos_id)
        elif isinstance(eos_id, int):
            self.stop_token_ids.add(eos_id)
            
        # 2. Collect from model generation_config
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            model_eos = getattr(self.model.generation_config, "eos_token_id", None)
            if isinstance(model_eos, list):
                self.stop_token_ids.update(model_eos)
            elif isinstance(model_eos, int):
                self.stop_token_ids.add(model_eos)
                
        # 3. Universally scan tokenizer's special tokens for genuine end-of-turn markers.
        # IMPORTANT: "<|im_start|>" is intentionally EXCLUDED — it is a message START
        # marker, not a stop signal. Including it causes premature generation halt when
        # the model predicts a next-turn prefix (e.g. structured output or roleplay).
        special_words = ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
        for word in special_words:
            tok_id = self.tokenizer.convert_tokens_to_ids(word)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tok_id)
                
        # 4. Fallback standard token
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                self.stop_token_ids.update(self.tokenizer.eos_token_id)
            else:
                self.stop_token_ids.add(self.tokenizer.eos_token_id)
                
        print(f"[DiffKV] Universal Stop Token IDs initialized: {sorted(list(self.stop_token_ids))}")
        
        # Apply Differential KV Attention Interception!
        apply_diffkv_attention_patch(self.model, self.manager)

        # ── Torch Compile JIT Fusion (auto-enabled for non-quantized models) ──
        # dynamic=True: handles variable chunk sizes without recompilation.
        # mode="reduce-overhead": enables horizontal kernel fusion (RMSNorm+SiLU+linear)
        #   which gives ~30-40% prefill throughput improvement.
        # For bitsandbytes/GPTQ/AWQ quantized models: skip compile — the custom quantized
        #   Linear layers cause graph breaks that prevent useful compilation.
        # On Windows: TorchInductor requires cl.exe (MSVC). Skip if not available.
        use_compile = "1" if self.manager.config.torch_compile else "0"
        if _is_apple_silicon():
            # macOS/MPS: torch.compile (even FFN-only compilation) via "aot_eager" or "inductor" backend 
            # introduces severe python tracing overhead which degrades generation throughput.
            # Thus, we disable it by default on Apple Silicon.
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
        if hasattr(self, "_cuda_graph_runner"):
            self._cuda_graph_runner.invalidate()

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

        for chunk_start in range(0, total_new, PREFILL_CHUNK):
            chunk_end = min(chunk_start + PREFILL_CHUNK, total_new)
            chunk = new_ids_list[chunk_start:chunk_end]
            abs_start = cached_len + chunk_start  # absolute position in full sequence

            chunk_tensor = torch.tensor([chunk], dtype=torch.long, device=self.device)
            pos_tensor = torch.arange(
                abs_start, abs_start + len(chunk),
                dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # Finalize any completed CPU background compressions from the previous chunk
            if hasattr(self.manager, "finalize_compressed_blocks"):
                self.manager.finalize_compressed_blocks()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=chunk_tensor,
                    position_ids=pos_tensor,
                    use_cache=True,
                )

            # Kick off async background SVD for this chunk immediately —
            # the next chunk's forward pass runs in parallel with SVD.
            if hasattr(self.manager, "compress_prefill_kv"):
                self.manager.compress_prefill_kv(session_id)

        # ── Post-prefill compression barrier ────────────────────────────────
        # Drain all background SVD results to the native pool before decode starts.
        # Since capture_prefill_kv() now streams immediately, all blocks have been
        # SUBMITTED. We just need to wait for the async compressor to mark them
        # CPU_COMPRESSED and then finalize (GPU upload) on the main thread.
        # Timeout: 30 s for very long prompts (research paper, code dumps, etc.)
        if hasattr(self.manager, "finalize_compressed_blocks"):
            _barrier_deadline = _time.monotonic() + 30.0
            while _time.monotonic() < _barrier_deadline:
                pending = getattr(self.manager, "_pending_cpu_blocks", 0)
                if pending <= 0:
                    break
                self.manager.finalize_compressed_blocks()
                _time.sleep(0.002)

        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [1, vocab]

        # CRITICAL FIX: track the absolute sequence position for each decode step.
        cur_pos = cached_len + prefill_len

        for _ in range(max_new_tokens):
            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok_id in set(generated[-64:]):
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
                            logits[0, tok_id] /= repetition_penalty
                        else:
                            logits[0, tok_id] *= repetition_penalty

            # Sample
            next_id = _compiled_sample_fn(logits, temperature, top_p)

            generated.append(next_id.item())
            if next_id.item() in self.stop_token_ids:
                break

            # Pass the correct absolute position so RoPE rotates at the right angle.
            # past_key_values is always None (DiffKV manages KV internally), so without
            # this the model would wrongly use position 0 for every decode token.
            pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=self.device)
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
                if not hasattr(self, "_cuda_graph_runner"):
                    self._cuda_graph_runner = CUDAGraphDecodeRunner() if _HAS_CUDA_GRAPH_RUNNER else None

                def _decode_model_fn(input_ids, position_ids, past_key_values, use_cache):
                    return self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                try:
                    outputs = self._cuda_graph_runner.run(
                        _decode_model_fn,
                        {
                            "input_ids":      input_ids,
                            "position_ids":   pos_tensor,
                            "past_key_values": past_kv,
                            "use_cache":      torch.tensor(True),  # static scalar
                        }
                    )
                except Exception:
                    # Graph capture failed (e.g. dynamic Python branching) — fall back to eager
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

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def _custom_sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
