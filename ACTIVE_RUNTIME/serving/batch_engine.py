import asyncio
import gc
import os
import re
import time
import threading
import torch
from collections import Counter
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer

class BatchRequest:
    def __init__(self, session_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float, repetition_penalty: float):
        self.session_id = session_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.prompt_ids: List[int] = []
        self.generated_ids: List[int] = []
        self.is_prefilled = False
        self.is_finished = False
        self.chunks_queue = asyncio.Queue()
        self.buffer = []
        self.first_token_time = None
        self.start_time = time.time()
        self.cancelled = False
        self.decoded_text = ""
        self.prefill_offset = 0
        # ── Repetition-loop detection state ──────────────────────────────
        # Tracks whether a token n-gram repetition loop has been detected.
        # When True, _sample switches to a wider penalty window (256 tokens)
        # and _emit_token terminates generation early.
        self.repetition_loop_detected: bool = False
        self._ngram_window: List[int] = []  # rolling window for n-gram counts

    @property
    def total_seq_len(self) -> int:
        """Total tokens seen so far = prompt + all generated tokens."""
        return len(self.prompt_ids) + len(self.generated_ids)

# ---------------------------------------------------------------------------
# Reference-list normalizer (Fix 1)
# ---------------------------------------------------------------------------

def _normalize_references(text: str) -> str:
    """Normalise citation-list formatting inconsistencies produced by the model.

    The model occasionally mixes styles in a single reference list:
      * [1] Author ...         <- bullet prefix
      [2] Author ...           <- clean
      In [3], Author ...       <- "In [N]," prefix

    This function rewrites every entry in the reference block to the clean
    "[N] Author ..." style, removing leading bullets and "In [N]," prefixes
    so the list is visually consistent.  The body text outside the reference
    section is left completely untouched.
    """
    lines = text.split('\n')
    
    # 1. Search for a reference header line
    header_re = re.compile(r'\b(references?|bibliography|works\s+cited|reference\s+list|sources|citations)\b', re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if len(line) <= 100 and header_re.search(line):
            header_idx = i
            # Keep searching to find the last/most relevant header
    
    # 2. Find matching reference entries
    ref_entry_re = re.compile(r'^(?:[iI]n\s+)?(?:[*\-•]\s*)?\[\d+\]')
    unambiguous_re = re.compile(r'^(?:[*\-•]\s*)?\[\d+\]')
    
    matching_indices = []
    unambiguous_indices = []
    for i, line in enumerate(lines):
        # If header found, only look after the header
        if header_idx is not None and i <= header_idx:
            continue
        stripped = line.strip()
        if ref_entry_re.match(stripped):
            matching_indices.append(i)
            if unambiguous_re.match(stripped):
                unambiguous_indices.append(i)
                
    # If a header was found but no matching entries after it, check the whole text
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
        
    # Determine the start of the reference block.
    if header_idx is not None:
        ref_start_idx = header_idx + 1
    elif unambiguous_indices:
        # Start at the first unambiguous entry
        ref_start_idx = unambiguous_indices[0]
    else:
        # If no header and no unambiguous entries, do not normalize
        return text
                
    body = '\n'.join(lines[:ref_start_idx])
    ref_block = '\n'.join(lines[ref_start_idx:])
    
    pattern = re.compile(
        r'^\s*'                   # leading spaces
        r'(?:[iI]n\s+)?'          # optional "In" or "in"
        r'(?:[*\-•]\s*)?'         # optional bullet
        r'(\[\d+\])'              # captured citation [N]
        r'(?:,\s*|\.\s*|\s+)?',   # optional comma, period, spaces after the citation
        re.MULTILINE
    )
    normalized_ref_block = pattern.sub(r'\1 ', ref_block)
    
    if body:
        return body + '\n' + normalized_ref_block
    return normalized_ref_block



# ---------------------------------------------------------------------------
# Repetition-loop detector (Fix 2)
# ---------------------------------------------------------------------------

# Minimum number of generated tokens before we bother checking for loops.
_LOOP_CHECK_MIN_TOKENS = 30
# N-gram size used for loop detection.
_LOOP_NGRAM_N = 5
# If the most-common n-gram occupies more than this fraction of the window,
# we declare a repetition loop.
_LOOP_NGRAM_THRESHOLD = 0.35
# How many of the most recent tokens we inspect for the n-gram analysis.
_LOOP_NGRAM_WINDOW = 80


def _detect_repetition_loop(generated_ids: List[int]) -> bool:
    """Return True if the recent token stream shows a repetition loop.

    We extract all n-grams from the tail of the generated sequence and
    flag a loop when a single n-gram dominates ≥35 % of the window.  This
    is robust against:
      - exact token-level loops (e.g. ero ero ero ...)
      - near-repeating patterns with slight variation
    """
    n = len(generated_ids)
    if n < _LOOP_CHECK_MIN_TOKENS:
        return False
    window = generated_ids[-_LOOP_NGRAM_WINDOW:]
    if len(window) < _LOOP_NGRAM_N + 1:
        return False
    ngrams = [
        tuple(window[i:i + _LOOP_NGRAM_N])
        for i in range(len(window) - _LOOP_NGRAM_N + 1)
    ]
    counts = Counter(ngrams)
    most_common_count = counts.most_common(1)[0][1]
    ratio = most_common_count / len(ngrams)
    return ratio >= _LOOP_NGRAM_THRESHOLD


@torch.jit.script
def _sample_gpu_jit(
    logits: torch.Tensor,                # [1, vocab_size]
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    generated_ids: torch.Tensor,        # [N_gen]
    prompt_ids: torch.Tensor,           # [N_prompt] — last 512 unique prompt tokens
) -> torch.Tensor:
    # 1. Repetition penalty — applied over BOTH generated tokens AND prompt tokens.
    # Without penalizing prompt tokens, the first generated token after a long prefill
    # has no diversity pressure and will freely pick the most frequent token in the
    # 6000-token context (which is always a word from the paper → recitation).
    if repetition_penalty != 1.0:
        vocab_size = logits.shape[-1]
        # Merge generated + prompt penalty sets. Generated IDs take priority.
        parts = []
        if generated_ids.numel() > 0:
            parts.append(generated_ids)
        if prompt_ids.numel() > 0:
            parts.append(prompt_ids)
        if parts:
            combined = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            penalty_ids = torch.unique(combined)
            penalty_ids = penalty_ids[penalty_ids < vocab_size]
            if penalty_ids.numel() > 0:
                scores = logits[0, penalty_ids]
                # Standard repetition-penalty formula
                scores = torch.where(scores > 0.0, scores / repetition_penalty, scores * repetition_penalty)
                logits[0].scatter_(0, penalty_ids, scores)

    if temperature <= 0.01:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = torch.nan_to_num(logits, nan=-100.0, posinf=100.0, neginf=-100.0)
    
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    probs_sum = probs.sum(dim=-1, keepdim=True)
    if (probs_sum == 0.0).any():
        probs = torch.ones_like(probs) / float(probs.shape[-1])
    else:
        probs = probs / probs_sum

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumulative - sorted_probs) > top_p
        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        psum = sorted_probs.sum(dim=-1, keepdim=True)
        sorted_probs = sorted_probs / torch.where(psum == 0.0, torch.ones_like(psum), psum)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)

    return torch.multinomial(probs, num_samples=1)


class CUDAStreamManager:
    """
    Manages CUDA streams for concurrent execution of prefill and decode tasks.
    Uses high-priority stream for decode to minimize latency, and low-priority for prefill.
    """
    def __init__(self):
        self.device_has_cuda = torch.cuda.is_available()
        if self.device_has_cuda:
            try:
                self.decode_stream = torch.cuda.Stream(priority=-1)
                self.prefill_stream = torch.cuda.Stream(priority=0)
            except Exception:
                self.decode_stream = torch.cuda.Stream()
                self.prefill_stream = torch.cuda.Stream()
        else:
            self.decode_stream = None
            self.prefill_stream = None

    def get_decode_stream(self):
        return self.decode_stream if self.device_has_cuda else None

    def get_prefill_stream(self):
        return self.prefill_stream if self.device_has_cuda else None


class ContinuousBatchEngine:
    def __init__(self, wrapper, max_batch_size=8, draft_wrapper=None):
        self.wrapper = wrapper
        self.cuda_stream_manager = CUDAStreamManager()
        self.max_batch_size = max_batch_size
        self.draft_wrapper = draft_wrapper
        self.active_requests: List[BatchRequest] = []
        self.incoming_queue = asyncio.Queue()
        self.is_running = False
        self._loop_task = None

        if self.draft_wrapper is not None:
            from plugins.speculative import SpeculativeDecodingPlugin
            from plugins.diffkv_as_draft import DiffKVAsDraftPlugin
            draft_plugin = DiffKVAsDraftPlugin(self.draft_wrapper)
            self.speculative_decoder = SpeculativeDecodingPlugin(self.wrapper, draft_plugin)

        if torch.backends.mps.is_available():
            try:
                import os
                cfg = getattr(self.wrapper.manager, "config", None)
                watermark = cfg.mps_watermark if cfg is not None else 0.0
                if cfg is not None:
                    approx = "1" if cfg.approximate_attn else "0"
                    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = approx
                else:
                    if os.environ.get("DIFFKV_MPS_APPROXIMATE_ATTN") is None:
                        os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(watermark)
                if watermark > 0:
                    torch.mps.set_per_process_memory_fraction(watermark)
            except Exception as e:
                print(f"[DiffKV] WARNING: Failed to set MPS memory fraction: {e}")

        self.tokenizer = self.wrapper.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._alphanumeric_tokens = {}
        self.session_token_ids = {}
        
        # Universal stop tokens inherited from wrapper
        self.stop_token_ids = getattr(self.wrapper, "stop_token_ids", {self.tokenizer.eos_token_id})
        
        # Track decode steps for periodic memory sweeps
        self.decode_steps_since_gc = 0
        self._prefill_input_buf = None
        self._prefill_pos_buf   = None

        # Track activity and idle timeouts (similar to Ollama)
        self.last_active_time = time.time()
        try:
            self.idle_timeout_seconds = float(os.environ.get("DIFFKV_MODEL_IDLE_TIMEOUT", "300"))
        except Exception:
            self.idle_timeout_seconds = 300.0
        print(f"[DiffKV] Idle model unloading configured for {self.idle_timeout_seconds} seconds of inactivity.")

    # ── VRAM instrumentation ────────────────────────────────────────────

    def _log_vram(self, tag: str):
        """Log VRAM stats (allocated and reserved) under a human-readable tag.
        Only active when DIFFKV_TELEMETRY=1 to avoid overhead in production."""
        if os.environ.get("DIFFKV_TELEMETRY", "0") != "1":
            return
        if not torch.cuda.is_available():
            return
        alloc_gb = torch.cuda.memory_allocated() / 1024 ** 3
        resv_gb  = torch.cuda.memory_reserved()   / 1024 ** 3
        print(f"[DiffKV VRAM] [{tag}] allocated={alloc_gb:.3f} GB  reserved={resv_gb:.3f} GB")

    def _post_prefill_cleanup(self):
        """
        Run once after prefill + GPU SVD compression completes.
        Releases any allocator-held staging buffers.
        """
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            # Safe Apple Silicon / MPS fallback
            _mps = getattr(torch, "mps", None)
            if _mps is not None:
                _empty = getattr(_mps, "empty_cache", None)
                _sync = getattr(_mps, "synchronize", None)
                if _empty is not None:
                    _empty()
                if _sync is not None:
                    _sync()
        
        if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved  = torch.cuda.memory_reserved()  / 1024**3
                print(f"[DiffKV] Post-prefill: {allocated:.2f}GB allocated, "
                      f"{reserved:.2f}GB reserved")
            else:
                print(f"[DiffKV] Post-prefill cleanup completed (non-CUDA platform).")

    # ── Compression barrier ─────────────────────────────────────────────

    async def _wait_for_compression(self, session_id: str, timeout_s: float = 30.0):
        """
        Non-blocking async wait until every block for *session_id* has left
        the SUBMITTED state (async SVD compressor is done), OR timeout elapses.

        CRITICAL: uses `await asyncio.sleep()` so the event loop is NEVER
        blocked. A sync `time.sleep()` here would freeze the entire server,
        causing SSE streams to idle-timeout and responses to appear truncated.
        """
        is_draft = session_id.endswith("_draft")
        wrapper = self.draft_wrapper if (is_draft and self.draft_wrapper is not None) else self.wrapper
        mgr = wrapper.manager
        streaming_mgr = getattr(mgr, "_streaming_mgr", None)
        if streaming_mgr is None:
            return

        session_blocks = streaming_mgr.session_blocks.get(session_id, {})
        if not session_blocks:
            return

        deadline = time.monotonic() + timeout_s
        check_interval = 0.005  # 5 ms async yield — event loop stays live

        while time.monotonic() < deadline:
            if hasattr(mgr, "finalize_compressed_blocks"):
                mgr.finalize_compressed_blocks()

            found_submitted = False
            for layer_idx, blocks in session_blocks.items():
                for block in blocks:
                    if getattr(block, "state", None) in ("SUBMITTED", "CPU_COMPRESSED"):
                        found_submitted = True
                        break
                if found_submitted:
                    break
            if not found_submitted:
                return  # All blocks compressed — safe to start decode
            await asyncio.sleep(check_interval)  # yield to event loop, stream chunks etc.

        # Timeout — warn but don’t block decode indefinitely
        if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            print(f"[DiffKV BatchEngine] WARNING: compression barrier timed out after {timeout_s}s "
                  f"for session {session_id}. Some blocks may still be SUBMITTED.")

    # ── SVD thread priority helpers ──────────────────────────────────────

    def _boost_compressor_priority(self):
        """Temporarily raise SVD worker thread priority during prefill burst.
        On Windows uses SetThreadPriority. On POSIX uses os.nice()."""
        compressor = getattr(self.wrapper.manager, "_compressor", None)
        if compressor is None:
            return
        workers = getattr(compressor, "_workers", [])
        if not workers:
            return
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                THREAD_PRIORITY_ABOVE_NORMAL = 1
                for t in workers:
                    handle = ctypes.windll.kernel32.OpenThread(0x0060, False, t.ident)  # THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION
                    if handle:
                        ctypes.windll.kernel32.SetThreadPriority(handle, THREAD_PRIORITY_ABOVE_NORMAL)
                        ctypes.windll.kernel32.CloseHandle(handle)
            else:
                # POSIX: renice to -5 (higher priority), clamped by OS permission
                for t in workers:
                    try:
                        os.setpriority(os.PRIO_PROCESS, t.ident, -5)
                    except PermissionError:
                        pass
        except Exception:
            pass  # Priority boost is best-effort

    def _restore_compressor_priority(self):
        """Restore SVD worker threads back to normal priority."""
        compressor = getattr(self.wrapper.manager, "_compressor", None)
        if compressor is None:
            return
        workers = getattr(compressor, "_workers", [])
        if not workers:
            return
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                THREAD_PRIORITY_NORMAL = 0
                for t in workers:
                    handle = ctypes.windll.kernel32.OpenThread(0x0060, False, t.ident)
                    if handle:
                        ctypes.windll.kernel32.SetThreadPriority(handle, THREAD_PRIORITY_NORMAL)
                        ctypes.windll.kernel32.CloseHandle(handle)
            else:
                for t in workers:
                    try:
                        os.setpriority(os.PRIO_PROCESS, t.ident, 0)
                    except PermissionError:
                        pass
        except Exception:
            pass

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._loop_task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
        
        # Cleanly stop wrapper background threads
        if hasattr(self, 'wrapper') and self.wrapper is not None:
            if hasattr(self.wrapper, 'stop'):
                self.wrapper.stop()
                
        # Purge class-level TritonDiffKV reconstruction buffers
        try:
            from native_core.sparse_decode.triton_fused_decode import TritonDiffKV
            if hasattr(TritonDiffKV, '_recon_buffers'):
                TritonDiffKV._recon_buffers.clear()
        except Exception as e:
            print(f"[DiffKV] Warning: failed to clear TritonDiffKV reconstruction buffers: {e}")

    async def submit(self, session_id: str, payload: Dict) -> asyncio.Queue:
        # Ensure model is loaded on demand when a request arrives
        if hasattr(self.wrapper, "ensure_loaded"):
            self.wrapper.ensure_loaded()
        if self.draft_wrapper is not None and hasattr(self.draft_wrapper, "ensure_loaded"):
            self.draft_wrapper.ensure_loaded()
        self.last_active_time = time.time()

        req = BatchRequest(
            session_id=session_id,
            prompt=payload["prompt"],
            max_tokens=payload.get("max_tokens", 16384),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.9),
            repetition_penalty=payload.get("repetition_penalty", 1.15)
        )

        encoded = self.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False)
        req.prompt_ids = encoded.input_ids[0].tolist()

        # O(1) Smart Prefix Check: check if the session already has resident KV cache.
        # If so, mark the cached length so prefill is incremental (avoiding O(N) re-prefill of history).
        #
        # IMPORTANT: session_token_ids[session_id] stores the FULL prompt token IDs from the
        # previous turn (chat-template-rendered), NOT prompt_ids + generated_ids concatenated.
        # This is because the chat template wraps assistant responses in special tokens
        # (<|im_start|>assistant\n...\n<|im_end|>), so raw concat does NOT match the new prompt.
        # The registry is updated at the end of each turn via update_session_token_prefix().
        req.cached_len = 0
        if hasattr(self.wrapper.manager, "get_session_sequence_length"):
            cached_len = self.wrapper.manager.get_session_sequence_length(session_id)
            if cached_len > 0 and cached_len < len(req.prompt_ids):
                stored_ids = self.session_token_ids.get(session_id, [])
                # Compare the STORED prefix (which is the re-tokenized full context from the
                # previous turn) against the beginning of the current prompt.
                # We only need to verify up to cached_len tokens match.
                compare_len = min(cached_len, len(stored_ids))
                if compare_len > 0 and req.prompt_ids[:compare_len] == stored_ids[:compare_len]:
                    req.cached_len = cached_len
                    print(f"[DiffKV BatchEngine] Found cached history for session {session_id}: "
                          f"length {cached_len} tokens (verified {compare_len} token prefix). Reusing KV cache!")
                else:
                    mismatch_idx = -1
                    if compare_len > 0:
                        for idx in range(compare_len):
                            if idx >= len(req.prompt_ids) or idx >= len(stored_ids) or req.prompt_ids[idx] != stored_ids[idx]:
                                mismatch_idx = idx
                                break
                    print(f"[DiffKV BatchEngine] Prefix mismatch at token index {mismatch_idx}: "
                          f"new_prompt_token={req.prompt_ids[mismatch_idx] if mismatch_idx >= 0 else None}, "
                          f"stored_token={stored_ids[mismatch_idx] if mismatch_idx >= 0 else None}")
                    print(f"[DiffKV BatchEngine] Prefix mismatch details for session {session_id}: "
                          f"cached_len={cached_len}, stored={len(stored_ids)}, new_prompt={len(req.prompt_ids)}.")
                    
                    if mismatch_idx > 32:
                        print(f"[DiffKV BatchEngine] Partially rolling back session {session_id} to token index {mismatch_idx} instead of fully clearing.")
                        self.wrapper.manager.rollback_session(session_id, mismatch_idx, clear_srl=True)
                        if self.draft_wrapper is not None:
                            self.draft_wrapper.manager.rollback_session(session_id + "_draft", mismatch_idx, clear_srl=True)
                        req.cached_len = mismatch_idx
                        self.session_token_ids[session_id] = stored_ids[:mismatch_idx]
                    else:
                        print(f"[DiffKV BatchEngine] Clearing stale KV cache and re-prefilling from scratch.")
                        # The stored context diverged — clear so we get a fresh prefill
                        self._free_session_kv(session_id)
                        if self.draft_wrapper is not None:
                            self._free_session_kv(session_id + "_draft", is_draft=True)

            # Token-level prefix search fallback: if no match in the current session, search other active sessions
            if req.cached_len == 0:
                longest_match_len = 0
                best_sid = None
                for sid, stored_ids in self.session_token_ids.items():
                    if sid == session_id:
                        continue
                    
                    session_seq_len = self.wrapper.manager.get_session_sequence_length(sid)
                    limit = min(len(stored_ids), len(req.prompt_ids) - 1)
                    limit = min(limit, session_seq_len)
                    
                    if limit > 0:
                        match_len = 0
                        if req.prompt_ids[:limit] == stored_ids[:limit]:
                            match_len = limit
                        else:
                            for idx in range(limit):
                                if req.prompt_ids[idx] != stored_ids[idx]:
                                    break
                                match_len = idx + 1
                        
                        if match_len >= 32 and match_len > longest_match_len:
                            longest_match_len = match_len
                            best_sid = sid
                
                if best_sid is not None:
                    # Clear any existing KV blocks allocated for the destination session_id before cloning
                    self._free_session_kv(session_id)
                    if self.draft_wrapper is not None:
                        self._free_session_kv(session_id + "_draft", is_draft=True)
                    
                    # Clone the matching session's KV cache (zero-copy metadata cloning)
                    self.wrapper.manager.clone_session(best_sid, session_id)
                    if self.draft_wrapper is not None:
                        self.draft_wrapper.manager.clone_session(best_sid + "_draft", session_id + "_draft")
                    
                    # If the cloned session's KV cache is longer than the matched prefix length,
                    # roll it back to match length (e.g. if the matching session was stopped mid-generation).
                    cloned_len = self.wrapper.manager.get_session_sequence_length(session_id)
                    if cloned_len > longest_match_len:
                        print(f"[DiffKV BatchEngine] Cloned session {session_id} has length {cloned_len} tokens, "
                              f"but matched prefix is {longest_match_len} tokens. Rolling back cloned cache to match length.")
                        self.wrapper.manager.rollback_session(session_id, longest_match_len, clear_srl=True)
                        if self.draft_wrapper is not None:
                            self.draft_wrapper.manager.rollback_session(session_id + "_draft", longest_match_len, clear_srl=True)
                    
                    # Update local token registry
                    self.session_token_ids[session_id] = self.session_token_ids[best_sid][:longest_match_len]
                    req.cached_len = longest_match_len
                    print(f"[DiffKV BatchEngine] Auto-matched sharing prefix from session {best_sid}: matched length {longest_match_len} tokens. Cloned KV cache successfully!")

        await self.incoming_queue.put(req)
        return req.chunks_queue

    def cancel(self, session_id: str, free_kv: bool = True):
        """Mark requests for this session_id as cancelled and free their KV cache immediately."""
        cancelled_count = 0
        # Cancel active requests
        for req in self.active_requests:
            if req.session_id == session_id:
                req.cancelled = True
                cancelled_count += 1

        # Cancel queued requests
        try:
            for req in list(self.incoming_queue._queue):
                if req.session_id == session_id:
                    req.cancelled = True
                    cancelled_count += 1
        except Exception as e:
            print(f"[DiffKV] Warning: failed to scan incoming queue for cancellation: {e}")

        if cancelled_count > 0:
            print(f"[DiffKV] Cancelled {cancelled_count} request(s) for session: {session_id}")
            if free_kv:
                self._free_session_kv(session_id)

    async def _batch_loop(self):
        while self.is_running:
            # 1. Drain incoming queue into active requests
            while not self.incoming_queue.empty() and len(self.active_requests) < self.max_batch_size:
                req = await self.incoming_queue.get()
                self.active_requests.append(req)

            if not self.active_requests:
                # Sleep briefly while idle — longer sleep here since we have no work
                try:
                    req = await asyncio.wait_for(self.incoming_queue.get(), timeout=0.05)
                    self.active_requests.append(req)
                except asyncio.TimeoutError:
                    # Check if the model has been idle too long and should be unloaded to free VRAM
                    if self.wrapper.model is not None:
                        idle_time = time.time() - self.last_active_time
                        if idle_time > self.idle_timeout_seconds:
                            print(f"\n[DiffKV] Server has been idle for {idle_time:.1f} seconds. "
                                  f"Unloading model weights from VRAM to free resources...")
                            if hasattr(self.wrapper, "close"):
                                self.wrapper.close()
                            if self.draft_wrapper is not None and hasattr(self.draft_wrapper, "close"):
                                self.draft_wrapper.close()
                    continue

            # 2. Filter out cancelled requests, freeing their KV cache.
            # Finished requests are NOT cleared; their KV cache is kept resident
            # and managed cleanly by ProductionSessionManager!
            for req in self.active_requests:
                if req.cancelled:
                    self._free_session_kv(req.session_id)
            self.active_requests = [r for r in self.active_requests if not r.cancelled and not r.is_finished]

            if not self.active_requests:
                continue

            # Update activity timestamp during processing
            self.last_active_time = time.time()

            try:
                await self._step()
            except RuntimeError as e:
                error_msg = str(e)
                print(f"Error in batch step: {error_msg}")
                
                # Provide helpful guidance for MPS out of memory errors
                if "MPS backend out of memory" in error_msg:
                    print("\n" + "="*80)
                    print("[DiffKV] MPS OUT OF MEMORY ERROR")
                    print("="*80)
                    print("Your Apple Silicon GPU has exceeded its 4GB memory limit.")
                    print("\nQuick fixes:")
                    print("  1. Restart with --preset low (enables aggressive memory reduction)")
                    print("  2. Use --serving-mode lightweight (reduces pool allocation)")
                    print("  3. Reduce --rank to 16 or 8 (smaller compression footprint)")
                    print("  4. Add --load-in-4bit (reduces model weights by 70%)")
                    print("\nRecommended command:")
                    print("  python serving/openai_compatible_api_gateway.py \\")
                    print("    --model Qwen/Qwen2.5-0.5B-Instruct \\")
                    print("    --preset low \\")
                    print("    --serving-mode lightweight \\")
                    print("    --rank 16")
                    print("="*80 + "\n")
                
                import traceback
                traceback.print_exc()
                for req in self.active_requests:
                    req.chunks_queue.put_nowait({"error": error_msg, "is_final": True})
                    req.is_finished = True
                    self._free_session_kv(req.session_id)
                self.active_requests.clear()
            except Exception as e:
                print(f"Error in batch step: {e}")
                import traceback
                traceback.print_exc()
                for req in self.active_requests:
                    req.chunks_queue.put_nowait({"error": str(e), "is_final": True})
                    req.is_finished = True
                    self._free_session_kv(req.session_id)
                self.active_requests.clear()

            # Minimal yield to event loop — use 0 so we yield without an artificial delay
            await asyncio.sleep(0)

    def _free_session_kv(self, session_id: str, is_draft: bool = False):
        """Release the KV manager blocks for a completed session to free VRAM."""
        try:
            if session_id in self.session_token_ids:
                del self.session_token_ids[session_id]
            wrapper = self.draft_wrapper if is_draft else self.wrapper
            kv_mgr = wrapper.manager
            if hasattr(kv_mgr, 'clear_session'):
                kv_mgr.clear_session(session_id)
        except Exception as e:
            print(f"[DiffKV] WARNING: could not free KV for session {session_id}: {e}")

    def update_session_token_prefix(self, session_id: str, full_prompt: str) -> None:
        """
        Called by the gateway after each response completes.

        Stores the token IDs as the prefix reference for the NEXT turn's prefix
        match in submit().

        The primary source of truth is _emit_token(), which sets
        session_token_ids[session_id] = req.prompt_ids + req.generated_ids
        immediately when the EOS token is generated. Those are the exact decode-time
        token IDs that match the KV block contents with 100% accuracy.

        This function provides a fallback retokenization for sessions that did not
        go through _emit_token (e.g. prefill-only or speculative decoder paths), and
        also updates kv_manager._session_token_ids for SRL index consistency.
        """
        try:
            # ── Primary: trust _emit_token's direct token ID store ──
            existing = self.session_token_ids.get(session_id)
            if existing:
                # _emit_token already stored exact prefix_ids — keep it.
                # Also update kv_manager._session_token_ids for SRL consistency.
                kv_mgr = getattr(self.wrapper, "manager", None)
                if kv_mgr is not None:
                    import torch as _torch
                    _sid_dict = getattr(kv_mgr, "_session_token_ids", None)
                    if _sid_dict is not None:
                        _sid_dict[session_id] = _torch.tensor(existing, dtype=_torch.long)
                print(f"[DiffKV BatchEngine] Prefix registry confirmed for session {session_id}: "
                      f"{len(existing)} tokens (from exact decode-time token IDs).")
                return

            # ── Fallback: retokenize (e.g. for sessions without decode phase) ──
            encoded = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
            token_ids = encoded.input_ids[0].tolist()
            self.session_token_ids[session_id] = token_ids

            kv_mgr = getattr(self.wrapper, "manager", None)
            if kv_mgr is not None:
                import torch as _torch
                _sid_dict = getattr(kv_mgr, "_session_token_ids", None)
                if _sid_dict is not None:
                    _sid_dict[session_id] = _torch.tensor(token_ids, dtype=_torch.long)

            print(f"[DiffKV BatchEngine] Updated prefix registry for session {session_id}: "
                  f"{len(token_ids)} tokens (fallback retokenization — no exact decode IDs found).")
        except Exception as e:
            print(f"[DiffKV BatchEngine] WARNING: failed to update prefix registry for {session_id}: {e}")


    async def _step(self):
        step_start = time.time()

        # Partition into PREFILL and DECODE
        prefill_reqs = [r for r in self.active_requests if not r.is_prefilled]
        decode_reqs  = [r for r in self.active_requests if r.is_prefilled]

        if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            if prefill_reqs:
                req0 = prefill_reqs[0]
                offset = getattr(req0, "prefill_offset", 0)
                cached = getattr(req0, "cached_len", 0)
                print(f"[DiffKV Step] PREFILL session={req0.session_id[:8]}... "
                      f"cached_len={cached} offset={offset}/{len(req0.prompt_ids)} "
                      f"({'Turn2+' if cached > 0 else 'Turn1'})")
        # ─────────────────────────────────────────────────────────────────
        # 1. CHUNKED PREFILL — process exactly one chunk of the first prefill request
        # ─────────────────────────────────────────────────────────────────
        if prefill_reqs:
            req = prefill_reqs[0]
            t0_pref = time.perf_counter()
            
            cached_len = getattr(req, "cached_len", 0)
            if not hasattr(req, "prefill_offset") or req.prefill_offset == 0:
                req.prefill_offset = cached_len
                # Fresh prefill from scratch — clear stale KV
                if cached_len == 0:
                    self._free_session_kv(req.session_id)
                    if self.draft_wrapper is not None:
                        self._free_session_kv(req.session_id + "_draft", is_draft=True)
            
            # Ensure session is registered and metadata initialized at the start of prefill
            if req.prefill_offset == cached_len:
                max_expected = len(req.prompt_ids) + req.max_tokens + 512
                if hasattr(self.wrapper.manager, "init_session"):
                    self.wrapper.manager.init_session(req.session_id, prefill_len=len(req.prompt_ids), max_tokens_hint=max_expected)
                if self.draft_wrapper is not None:
                    self.draft_wrapper.manager.init_session(req.session_id + "_draft", prefill_len=len(req.prompt_ids), max_tokens_hint=max_expected)

            # Inject session ID so the attention patch stores KV under the right key
            self.wrapper.model._diffkv_session_ids = [req.session_id]

            # Chunk size strategy:
            # • First turn (cached_len == 0): process the ENTIRE prompt in one forward pass.
            #   Cross-chunk LSE combination has a causal-masking inconsistency — lse_local is
            #   computed over unmasked scores while out_local uses causal-masked SDPA. For any
            #   multi-chunk first-turn prefill this produces garbage output. One big chunk has
            #   perfect standard causal attention with no LSE combination needed.
            #   Cap at 16384 to avoid OOM on extremely long inputs.
            # • Subsequent turns (cached_len > 0): only the NEW delta tokens are prefilled
            #   (always << 2048), so chunking is never triggered anyway.
            remaining = len(req.prompt_ids) - req.prefill_offset
            if cached_len == 0:
                # First turn: process the prompt in chunks.
                # Cap at 2048 on MPS to prevent prefill memory spikes, otherwise 16384.
                is_mps = (self.wrapper.device == "mps" or 
                          (isinstance(self.wrapper.device, torch.device) and self.wrapper.device.type == "mps"))
                
                # Check if DiffKV is bypassed for this session (length < engage threshold)
                from runtime.diffkv_attention import _get_engage_threshold
                is_bypassed = len(req.prompt_ids) < _get_engage_threshold()
                
                if is_bypassed:
                    max_chunk = 16384
                else:
                    env_chunk = os.environ.get("DIFFKV_PREFILL_CHUNK_SIZE")
                    if env_chunk is not None:
                        max_chunk = int(env_chunk)
                    else:
                        cfg = getattr(self.wrapper.manager, "config", None)
                        if cfg is not None:
                            max_chunk = cfg.prefill_chunk_size
                        elif is_mps:
                            max_chunk = 512
                        else:
                            max_chunk = 2048
                chunk_size = min(remaining, max_chunk)
            else:
                cfg = getattr(self.wrapper.manager, "config", None)
                if cfg is not None:
                    chunk_size = min(remaining, cfg.prefill_chunk_size)
                elif self.wrapper.device == "mps" or (isinstance(self.wrapper.device, torch.device) and self.wrapper.device.type == "mps"):
                    chunk_size = min(remaining, 512)
                else:
                    chunk_size = 2048
            offset = req.prefill_offset
            chunk_ids = req.prompt_ids[offset : offset + chunk_size]
            actual_len = len(chunk_ids)
            is_last_chunk = (offset + actual_len >= len(req.prompt_ids))
            # Track chunk index for periodic MPS flush (prevents Metal cmd buffer accumulation)
            _prefill_chunk_idx = offset // chunk_size if chunk_size > 0 else 0


            # Lazy pre-allocation of reusable input + position buffers (one allocation for entire prefill)
            _use_pinned = (self.wrapper.device == "cuda" or
                           (isinstance(self.wrapper.device, torch.device) and self.wrapper.device.type == "cuda"))
            if self._prefill_input_buf is None or self._prefill_input_buf.shape[1] < chunk_size:
                if _use_pinned:
                    self._prefill_input_buf = torch.zeros((1, chunk_size), dtype=torch.long).pin_memory()
                else:
                    self._prefill_input_buf = torch.zeros((1, chunk_size), dtype=torch.long)
            if not hasattr(self, '_prefill_pos_buf') or self._prefill_pos_buf is None or self._prefill_pos_buf.shape[1] < chunk_size:
                self._prefill_pos_buf = torch.zeros((1, chunk_size), dtype=torch.long)

            # In-place fill of reusable buffers — zero new Python objects per chunk
            self._prefill_input_buf[0, :actual_len] = torch.as_tensor(chunk_ids, dtype=torch.long)
            input_ids = self._prefill_input_buf[:, :actual_len].to(self.wrapper.device, non_blocking=True)

            # Re-use position buffer in-place instead of torch.arange() per chunk
            self._prefill_pos_buf[0, :actual_len] = torch.arange(offset, offset + actual_len, dtype=torch.long)
            position_ids = self._prefill_pos_buf[:, :actual_len].to(self.wrapper.device, non_blocking=True)

            # Fix 3: Only flush completed CPU compressions on the LAST chunk.
            # Previously this was called every chunk, blocking each chunk against the
            # background SVD thread and adding latency proportional to N_chunks.
            # On intermediate chunks the compressed blocks are not needed yet (the
            # incremental prefill path also reads uncompressed dense blocks fine).
            if is_last_chunk and hasattr(self.wrapper.manager, "finalize_compressed_blocks"):
                self.wrapper.manager.finalize_compressed_blocks()

            # ── SRL: register token IDs for this chunk before forward pass ────
            if hasattr(self.wrapper.manager, "register_prefill_tokens"):
                # Re-use the already-filled _prefill_input_buf slice (CPU) — no extra allocation
                chunk_tensor_cpu = self._prefill_input_buf[0, :actual_len]
                self.wrapper.manager.register_prefill_tokens(
                    req.session_id, chunk_tensor_cpu
                )
                if self.draft_wrapper is not None:
                    self.draft_wrapper.manager.register_prefill_tokens(
                        req.session_id + "_draft", chunk_tensor_cpu
                    )

            prefill_stream = self.cuda_stream_manager.get_prefill_stream()
            if prefill_stream is not None:
                with torch.cuda.stream(prefill_stream):
                    if self.draft_wrapper is not None:
                        draft_session_id = req.session_id + "_draft"
                        draft_input_ids = input_ids.to(self.draft_wrapper.device)
                        draft_position_ids = position_ids.to(self.draft_wrapper.device)
                        self.draft_wrapper.model._diffkv_session_ids = [draft_session_id]
                        with torch.no_grad():
                            self.draft_wrapper.model(
                                input_ids=draft_input_ids,
                                position_ids=draft_position_ids,
                                use_cache=True
                            )
                        if hasattr(self.draft_wrapper.manager, "compress_prefill_kv"):
                            self.draft_wrapper.manager.compress_prefill_kv(draft_session_id)
                    with torch.no_grad():
                        out = self.wrapper.model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            use_cache=True
                        )
            else:
                if self.draft_wrapper is not None:
                    draft_session_id = req.session_id + "_draft"
                    draft_input_ids = input_ids.to(self.draft_wrapper.device)
                    draft_position_ids = position_ids.to(self.draft_wrapper.device)
                    self.draft_wrapper.model._diffkv_session_ids = [draft_session_id]
                    with torch.no_grad():
                        self.draft_wrapper.model(
                            input_ids=draft_input_ids,
                            position_ids=draft_position_ids,
                            use_cache=True
                        )
                    if hasattr(self.draft_wrapper.manager, "compress_prefill_kv"):
                        self.draft_wrapper.manager.compress_prefill_kv(draft_session_id)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                        torch.mps.empty_cache()

                with torch.no_grad():
                    out = self.wrapper.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        use_cache=True
                    )

            req.prefill_offset += actual_len

            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                # Flush Metal command buffers every 4 chunks to prevent accumulation.
                # Without this, 32 chunks × ~80 MB Metal intermediates = ~2.5 GB driver overhead.
                # Every-4 balances RAM usage vs dispatch latency (4 chunks = 1024 tokens between flushes).
                if is_last_chunk or (_prefill_chunk_idx % 4 == 3):
                    torch.mps.empty_cache()
                    import gc as _gc; _gc.collect()  # return freed malloc arenas to the OS
                await asyncio.sleep(0)

            # position_ids was filled from a reusable buffer; only need to drop the device view
            del input_ids, position_ids

            # Double-buffered async compression after each chunk
            if hasattr(self.wrapper.manager, "compress_prefill_kv"):
                self.wrapper.manager.compress_prefill_kv(req.session_id)

            if req.prefill_offset >= len(req.prompt_ids):
                req.is_prefilled = True
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Fix 3: Trigger compression of all deferred prefill blocks now that prefill is complete.
                # This applies to ALL turns (Turn 1 AND Turn 2+), ensuring newly ingested blocks
                # from continuation turns get compressed promptly rather than accumulating as dense
                # blocks that grow the O(n) attention window with each turn.
                if hasattr(self.wrapper.manager, "compress_deferred_prefill_blocks"):
                    self.wrapper.manager.compress_deferred_prefill_blocks(req.session_id)
                if self.draft_wrapper is not None and hasattr(self.draft_wrapper.manager, "compress_deferred_prefill_blocks"):
                    self.draft_wrapper.manager.compress_deferred_prefill_blocks(req.session_id + "_draft")
                    
                logits = out.logits[:, -1, :]  # last token logits — first generated token
                next_id = self._sample(logits, req)
                req.generated_ids.append(next_id)
                self._emit_token(req, next_id, step_start)

                # Step 2: Release allocator-held staging buffers
                self._post_prefill_cleanup()
                self._log_vram(f"post-prefill session={req.session_id}")

                if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
                    dur_pref = (time.perf_counter() - t0_pref) * 1000
                    print(f"[DiffKV Telemetry] Prefill session={req.session_id} tokens={len(req.prompt_ids)} duration={dur_pref:.2f}ms")
                    if hasattr(self.wrapper.manager, "log_block_states"):
                        self.wrapper.manager.log_block_states(req.session_id)

                # Fix 4: Fire-and-forget compression + SRL index build.
                # The first token is already streamed above via _emit_token.
                # Compression and SRL index are built in background so they
                # don't block the decode loop.
                #
                # CRITICAL: finalize_srl_index is CPU-heavy sync code. Running it
                # directly inside an asyncio coroutine freezes the event loop and
                # prevents queue.get() from ever unblocking (causing 120s timeouts).
                # We use loop.run_in_executor (ThreadPoolExecutor) to offload it.
                _sid = req.session_id
                _mgr = self.wrapper.manager
                _draft_mgr = self.draft_wrapper.manager if self.draft_wrapper is not None else None
                _cached_len = getattr(req, "cached_len", 0)
                _loop = asyncio.get_event_loop()
                _is_first_turn = (_cached_len == 0)

                async def _build_srl_index_async():
                    _t_srl_start = time.perf_counter()

                    # ── Fix 2: Adaptive Turn 2+ SRL rebuild threshold ──────────
                    # When cached_len > 0, this is a continuation turn. The new delta
                    # tokens generate new ACCUMULATING blocks per layer. We skip the
                    # expensive compression barrier+rebuild by default, but we DO rebuild
                    # when the block count has grown by >20% since the last build —
                    # otherwise the SRL router permanently routes to stale Turn-1 blocks
                    # and quality degrades badly in long sessions (observed in telemetry:
                    # session grew from 6→12 blocks but router kept returning Turn-1 slots).
                    if not _is_first_turn:
                        srl_state_existing = _mgr.get_srl_state(_sid)
                        if srl_state_existing is not None:
                            n_current = srl_state_existing.n_active_blocks()
                            # n_blocks_at_build is set whenever we finish building the index
                            n_at_build = getattr(srl_state_existing, "n_blocks_at_build", n_current)
                            growth_ratio = (n_current - n_at_build) / max(1, n_at_build)
                            if growth_ratio < 0.20:
                                # < 20% growth — fast path, skip rebuild
                                print(f"[DiffKV BatchEngine] Turn 2+: SRL index already valid for session {_sid} "
                                      f"({n_current} blocks, built at {n_at_build}, growth={growth_ratio:.0%}). "
                                      f"Skipping compression barrier and SRL rebuild. "
                                      f"(saved ~{n_current * 28 // 1000:.1f}k SVD ops)")
                                return  # ← decode starts immediately, no wait
                            else:
                                # ≥ 20% growth — rebuild needed for accurate routing
                                print(f"[DiffKV BatchEngine] Turn 2+: SRL index stale for session {_sid} "
                                      f"({n_current} blocks vs {n_at_build} at build, growth={growth_ratio:.0%}). "
                                      f"Triggering incremental SRL rebuild.")

                    # ── First turn: wait for compression then build SRL ──────────────
                    print(f"[DiffKV BatchEngine] First-turn SRL build: waiting for compression barrier...")
                    _t_barrier_start = time.perf_counter()
                    # 1. Wait for SVD compression to finish (async — yields to event loop)
                    await self._wait_for_compression(_sid)
                    if _draft_mgr is not None:
                        await self._wait_for_compression(_sid + "_draft")
                    _t_barrier_end = time.perf_counter()
                    print(f"[DiffKV BatchEngine] Compression barrier done in {(_t_barrier_end - _t_barrier_start)*1000:.1f}ms")

                    # 2. Build SRL index in a thread so the event loop stays live
                    def _do_finalize():
                        if hasattr(_mgr, "finalize_srl_index"):
                            _mgr.finalize_srl_index(_sid, cached_len=_cached_len)
                        if _draft_mgr is not None and hasattr(_draft_mgr, "finalize_srl_index"):
                            _draft_mgr.finalize_srl_index(_sid + "_draft", cached_len=_cached_len)
                    try:
                        _t_finalize_start = time.perf_counter()
                        await _loop.run_in_executor(None, _do_finalize)
                        _t_finalize_end = time.perf_counter()
                        print(f"[DiffKV BatchEngine] SRL index built in {(_t_finalize_end - _t_finalize_start)*1000:.1f}ms "
                              f"| total={(_t_finalize_end - _t_srl_start)*1000:.1f}ms")

                        # ── Pre-warm SRL routing for the first decode step ──
                        srl_state = _mgr.get_srl_state(_sid)
                        if srl_state is not None:
                            # Stamp the block count at build time so the Turn 2+ growth
                            # threshold check (Fix 2) knows when a rebuild is warranted.
                            srl_state.n_blocks_at_build = srl_state.n_active_blocks()
                        if srl_state is not None and getattr(srl_state, "last_prefill_q", None) is not None:
                            pool = getattr(_mgr, "native_pool", None)
                            if pool is not None:
                                from native_core.srl.query_router import route_query_fixed_k
                                import math
                                q_for_routing = srl_state.last_prefill_q
                                head_dim = q_for_routing.shape[-1]
                                _scale = 1.0 / math.sqrt(head_dim)
                                selected_slots = route_query_fixed_k(
                                    Q         = q_for_routing,
                                    srl_state = srl_state,
                                    pool      = pool,
                                    scale     = _scale,
                                    layer_idx = 0,
                                )
                                srl_state.current_step_slots = selected_slots
                                srl_state.current_step_count = 0
                                if os.environ.get("DIFFKV_SRL_VERBOSE", "0") == "1" or os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
                                    print(f"[SRL Pre-warm] Pre-warmed routing for session {_sid}: "
                                          f"selected {selected_slots.numel()}/{srl_state.n_active_blocks()} blocks")

                    except Exception as _e:
                        print(f"[DiffKV BatchEngine] WARNING: SRL index build/pre-warm failed: {_e}")
                        import traceback
                        traceback.print_exc()
                        pass  # SRL index failure is non-fatal; decode continues without routing

                asyncio.ensure_future(_build_srl_index_async())

        # ─────────────────────────────────────────────────────────────────
        # 2. BATCHED DECODE (B >= 1)
        # CUDA: uses CUDAGraphDecodeRunner for ~2µs graph replay overhead.
        # MPS:  runs eager — fused_decode_attention_mps fires automatically
        #       inside the DiffKV attention patch (triton_sparse_attn.py).
        # ─────────────────────────────────────────────────────────────────
        if decode_reqs:
            # Finalize any completed async compressions before decode
            if hasattr(self.wrapper.manager, "finalize_compressed_blocks"):
                self.wrapper.manager.finalize_compressed_blocks()

            if self.draft_wrapper is not None and len(decode_reqs) == 1:
                # ── Speculative Decoding Fast Path (Single request) ─────────────
                req = decode_reqs[0]
                t0_dec = time.perf_counter()
                new_tokens = self.speculative_decoder.run_step(
                    req, step_start, self._sample, self._emit_token
                )
                self.session_token_ids[req.session_id] = req.prompt_ids + req.generated_ids
                
                if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
                    dur_dec = (time.perf_counter() - t0_dec) * 1000
                    print(f"[DiffKV Telemetry] Speculative Decode Step candidates={self.speculative_decoder.num_candidates} "
                          f"accepted={len(new_tokens) - 1} dur={dur_dec:.2f}ms")
                
                # Check for periodic memory sweeps
                self.decode_steps_since_gc += 1
                if self.decode_steps_since_gc >= 100:
                    self.decode_steps_since_gc = 0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        _mps = getattr(torch, "mps", None)
                        if _mps is not None:
                            _empty = getattr(_mps, "empty_cache", None)
                            if _empty is not None:
                                _empty()
                return

            t0_dec = time.perf_counter()
            input_ids_list = []
            position_ids_list = []
            session_ids = []

            for req in decode_reqs:
                cur_pos = req.total_seq_len - 1
                input_ids_list.append([req.generated_ids[-1]])
                position_ids_list.append([cur_pos])
                session_ids.append(req.session_id)

            actual_batch_size = len(decode_reqs)

            # Power-of-2 bucket for shape stability (required for CUDA graph replay)
            bucket_size = 1
            while bucket_size < actual_batch_size:
                bucket_size *= 2

            # Pad to bucket with dummy rows if needed
            if actual_batch_size < bucket_size:
                dummy_req = decode_reqs[-1]
                for _ in range(bucket_size - actual_batch_size):
                    input_ids_list.append([dummy_req.generated_ids[-1]])
                    position_ids_list.append([dummy_req.total_seq_len - 1])
                    session_ids.append("dummy_session")

            input_ids  = torch.tensor(input_ids_list,  dtype=torch.long, device=self.wrapper.device)
            position_ids = torch.tensor(position_ids_list, dtype=torch.long, device=self.wrapper.device)

            # Inject session IDs into the model so the attention patch routes correctly
            self.wrapper.model._diffkv_session_ids = session_ids

            is_cuda = (self.wrapper.device == "cuda" or
                       (isinstance(self.wrapper.device, torch.device) and
                        self.wrapper.device.type == "cuda"))

            with torch.no_grad():
                # ── CUDA path: try CUDA graph runner first ──────────────────
                _ran_graph = False
                if is_cuda:
                    decode_stream = self.cuda_stream_manager.get_decode_stream()
                    if decode_stream is not None:
                        with torch.cuda.stream(decode_stream):
                            runner = getattr(self.wrapper, "_cuda_graph_runner", None)
                            if runner is not None and runner.is_captured():
                                try:
                                    out = runner.run(input_ids, position_ids)
                                    _ran_graph = True
                                except Exception as _ge:
                                    runner.invalidate()
                            if not _ran_graph:
                                out = self.wrapper.model(
                                    input_ids=input_ids,
                                    position_ids=position_ids,
                                    use_cache=True,
                                )
                                if runner is not None and not runner.is_captured():
                                    try:
                                        runner.capture(self.wrapper.model, input_ids, position_ids)
                                    except Exception:
                                        pass
                    else:
                        runner = getattr(self.wrapper, "_cuda_graph_runner", None)
                        if runner is not None and runner.is_captured():
                            try:
                                out = runner.run(input_ids, position_ids)
                                _ran_graph = True
                            except Exception as _ge:
                                runner.invalidate()
                        if not _ran_graph:
                            out = self.wrapper.model(
                                input_ids=input_ids,
                                position_ids=position_ids,
                                use_cache=True,
                            )
                            if runner is not None and not runner.is_captured():
                                try:
                                    runner.capture(self.wrapper.model, input_ids, position_ids)
                                except Exception:
                                    pass
                else:
                    # ── MPS / CPU path: run normally ────────────────────────
                    # fused_decode_attention_mps() fires automatically inside
                    # diffkv_attention.py when the session has compressed blocks
                    # and device == mps. Wrap in capture_to_graph to cache execution graph.
                    is_mps = (self.wrapper.device == "mps" or
                              (isinstance(self.wrapper.device, torch.device) and
                               self.wrapper.device.type == "mps"))
                    is_mlx = getattr(self.wrapper, "is_mlx", False)
                    if is_mps and not is_mlx and hasattr(torch, "mps") and hasattr(torch.mps, "capture_to_graph"):
                        with torch.mps.capture_to_graph():
                            out = self.wrapper.model(
                                input_ids=input_ids,
                                position_ids=position_ids,
                                use_cache=True,
                            )
                    else:
                        out = self.wrapper.model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            use_cache=True,
                        )

            logits = out.logits[:, -1, :]  # [bucket_size, vocab_size]

            # Extract and sample outputs ONLY for actual active requests
            for idx in range(actual_batch_size):
                req = decode_reqs[idx]
                req_logits = logits[idx : idx + 1]
                next_id = self._sample(req_logits, req)
                req.generated_ids.append(next_id)
                self._emit_token(req, next_id, step_start)
                # Store only prompt_ids as the prefix reference if the request is not finished.
                # If the request has finished, _emit_token() has already stored the exact full
                # prefix (prompt_ids + generated_ids). Overwriting it here would cause prefix mismatch.
                if not req.is_finished:
                    self.session_token_ids[req.session_id] = req.prompt_ids

            if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
                dur_dec = (time.perf_counter() - t0_dec) * 1000
                graph_tag = " [graph]" if _ran_graph else " [eager]"
                print(f"[DiffKV Telemetry] Decode Step batch={actual_batch_size} "
                      f"bucket={bucket_size} dur={dur_dec:.2f}ms{graph_tag}")



            self.decode_steps_since_gc += 1
            if self.decode_steps_since_gc >= 100:
                self.decode_steps_since_gc = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    _mps = getattr(torch, "mps", None)
                    if _mps is not None:
                        _empty = getattr(_mps, "empty_cache", None)
                        if _empty is not None:
                            _empty()

    def _sample(self, logits: torch.Tensor, req: BatchRequest) -> int:
        srl_state = self.wrapper.manager.get_srl_state(req.session_id)
        if len(req.generated_ids) == 0:
            req.sfa_active = False
            if srl_state is not None:
                srl_state.vsl_active_candidates = []
                srl_state.vsl_consecutive_helpers = 0
                # Reset the query anchor so each new response anchors to its own
                # first decode step, not the previous response's Q-vector.
                # Without this reset, all turns in a multi-turn chat are 35%
                # anchored toward turn-1's query, causing recurring retrieval of
                # the same entries regardless of the actual question.
                srl_state.factual_anchor_q = None

        # ── Repetition-loop recovery (Fix 2) ────────────────────────────────
        # When a loop has been detected we widen the penalty window from 64 to
        # 256 tokens and boost the penalty strength to break the loop faster.
        if req.repetition_loop_detected:
            penalty_window = 256
            penalty_val = max(req.repetition_penalty, 1.3)
        else:
            penalty_window = 64
            # Prompt anti-copy guard for the first 8 generated tokens.
            if req.prompt_ids and len(req.generated_ids) < 8:
                penalty_val = max(req.repetition_penalty, 1.15)
            else:
                penalty_val = req.repetition_penalty

        if req.generated_ids:
            gen_tensor = torch.tensor(
                req.generated_ids[-penalty_window:], dtype=torch.long, device=logits.device
            )
        else:
            gen_tensor = torch.empty((0,), dtype=torch.long, device=logits.device)

        # Prompt anti-copy guard: apply a prompt-token penalty on the first 8 generated
        # tokens. This guides the tiny model to start with its own words (e.g. "This paper...")
        # instead of immediately reciting the prompt. We use the last 512 prompt tokens
        # to cover the local context, and enforce an anti-copy penalty of at least 1.15
        # even if the request's repetition penalty is unset (1.0).
        if not req.repetition_loop_detected and req.prompt_ids and len(req.generated_ids) < 8:
            prompt_tensor = torch.tensor(
                req.prompt_ids[-512:], dtype=torch.long, device=logits.device
            )
        else:
            prompt_tensor = torch.empty((0,), dtype=torch.long, device=logits.device)

        # Apply Factual Logit Bias
        if srl_state is not None:
            # Helper token set (needed for penalty and VSL masking below)
            from native_core.srl.factual_alignment import get_helper_token_ids
            helper_ids = get_helper_token_ids(self.tokenizer)

            # +7.0 factual token bias (raised from +3)
            if getattr(srl_state, "current_step_factual_tokens", None):
                for tok_id in srl_state.current_step_factual_tokens:
                    if tok_id < logits.shape[-1]:
                        logits[0, tok_id] += 7.0

            # +7.0 VSL active-candidate boost
            active_candidates = getattr(srl_state, "vsl_active_candidates", [])
            if active_candidates:
                for suffix in active_candidates:
                    if suffix and suffix[0] < logits.shape[-1]:
                        logits[0, suffix[0]] += 7.0

            # -3.5 anti-hallucination penalty — only fires at sim ≥ 0.55 to avoid
            # penalising all domain vocabulary when a weak/wrong match passes the
            # lower factual-query threshold. At 0.3 almost every entry in a focused
            # document matches, so the "excluded" token set was nearly the full vocab
            # which makes the penalty useless and the factual set the only option.
            # Raising to 0.55 means we only penalise when we're genuinely confident.
            if (getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55
                    and getattr(srl_state, "current_step_factual_tokens", None)):
                factual_set = srl_state.current_step_factual_tokens
                _vocab = logits.shape[-1]
                _excl = [t for t in list(factual_set) + list(helper_ids) if 0 <= t < _vocab]
                if _excl:
                    _excl_t = torch.tensor(_excl, dtype=torch.long, device=logits.device)
                    _penalty_mask = torch.ones(_vocab, dtype=torch.bool, device=logits.device)
                    _penalty_mask.scatter_(0, _excl_t, False)
                    logits[0, _penalty_mask] -= 3.5

            # +10.0 transition bias — only fire when the last token is a CONTENT word
            # (not in helper_ids). Helper words (the, is, of, a, ...) appear at random
            # positions across many sequences; boosting their successors adds noise not
            # signal and drives the model into wrong sequence continuations.
            last_token = req.generated_ids[-1] if req.generated_ids else None
            if (last_token is not None
                    and last_token not in helper_ids
                    and getattr(srl_state, "current_step_factual_sequences", None)):
                transition_candidates = set()
                for seq in srl_state.current_step_factual_sequences:
                    for idx, tok in enumerate(seq[:-1]):
                        if tok == last_token:
                            transition_candidates.add(seq[idx + 1])
                for tok_id in transition_candidates:
                    if tok_id < logits.shape[-1]:
                        logits[0, tok_id] += 10.0

        # Apply Dynamic Temperature Scaling — threshold raised 0.3→0.55 to match
        # the tighter retrieval confidence bar.
        effective_temperature = req.temperature
        if srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55:
            max_sim = srl_state.current_step_max_similarity
            effective_temperature = req.temperature * (1.0 - max_sim * 0.95)

        # SFA threshold raised 0.3→0.55: at 0.3 almost every topical entry matches,
        # activating the VSL and forcing generation from a mixed-category token set.
        # At 0.55 only high-confidence, specific retrieval triggers the constraint.
        req.sfa_active = (
            srl_state is not None
            and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.55
            and bool(getattr(srl_state, "current_step_factual_sequences", None))
        )

        # LM-VSL (Logit Masking) — graduated by retrieval confidence:
        #
        #  sim 0.55–0.69 → soft penalty (-7): model can escape if its LM distribution
        #    is very strong; prevents total lock-in on weak retrievals.
        #  sim ≥ 0.70    → hard mask (-1e10): high-confidence retrieval forces verbatim
        #    extraction. With the new sequence-start-only fallback in get_allowed_tokens_vsl,
        #    the model must enter a factual sequence from its first token and advance
        #    through it in order — this is the fix for entity binding and relationship
        #    inversion (the model was picking correct keywords in wrong sentence structures).
        #
        # The VSL now uses sequence-start-only fallback when unlocked (only seq[0] of each
        # sequence is allowed when no lock is active), and the lock persists through 12
        # consecutive helpers instead of 4, so normal bridge phrases don't break it.
        if req.sfa_active:
            from native_core.srl.factual_alignment import get_allowed_tokens_vsl
            allowed_ids = get_allowed_tokens_vsl(srl_state, helper_ids)
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[list(allowed_ids)] = False
            max_sim = getattr(srl_state, "current_step_max_similarity", 0.0)
            if max_sim >= 0.70:
                logits[0, mask] = -1e10   # hard: verbatim extraction mode
            else:
                logits[0, mask] -= 7.0    # soft: guided but escapable

        sampled_tensor = _sample_gpu_jit(
            logits,
            effective_temperature,
            req.top_p,
            penalty_val,
            gen_tensor,
            prompt_tensor,
        )
        return sampled_tensor.item()

    def _emit_token(self, req: BatchRequest, token_id: int, step_start: float):
        if req.first_token_time is None:
            req.first_token_time = time.time()

        is_eos = (token_id in self.stop_token_ids)

        # Strict Factual Alignment (SFA) State Update and Loop Check
        srl_state = self.wrapper.manager.get_srl_state(req.session_id)
        if getattr(req, "sfa_active", False) and srl_state is not None:
            from native_core.srl.factual_alignment import update_vsl_state, get_helper_token_ids
            helper_ids = get_helper_token_ids(self.tokenizer)
            update_vsl_state(token_id, srl_state, helper_ids)
            
            if getattr(srl_state, "vsl_consecutive_helpers", 0) >= 16:
                if req.generated_ids:
                    req.generated_ids.pop()
                uncertainty_suffix = " [uncertain: details missing in source]"
                uncertainty_tokens = self.tokenizer.encode(uncertainty_suffix, add_special_tokens=False)
                req.generated_ids.extend(uncertainty_tokens)
                is_eos = True

        # Factual Early Stopping (Option 2 Extension)
        if not is_eos and srl_state is not None and getattr(srl_state, "current_step_max_similarity", 0.0) >= 0.5:
            if getattr(srl_state, "current_step_factual_sequences", None):
                for seq in srl_state.current_step_factual_sequences:
                    if len(seq) >= 5 and token_id == seq[-1]:
                        is_eos = True
                        break

        is_max = (len(req.generated_ids) >= req.max_tokens)

        # ── Repetition-loop detection (Fix 2) ────────────────────────────────
        # Check for n-gram loops every 10 tokens after the minimum warm-up period.
        # We only escalate once (flip the flag) to avoid spamming the check.
        if (not req.repetition_loop_detected
                and not is_eos
                and not is_max
                and len(req.generated_ids) % 10 == 0):
            if _detect_repetition_loop(req.generated_ids):
                req.repetition_loop_detected = True
                print(
                    f"[DiffKV] WARNING: repetition loop detected for session "
                    f"{req.session_id} at token {len(req.generated_ids)}. "
                    "Escalating penalty window to 256 tokens and strength to 1.3x."
                )

        # If a loop persists for more than 40 tokens after detection, terminate early.
        if req.repetition_loop_detected and not is_eos and not is_max:
            # Count how many tokens were generated after loop detection began.
            # We store the detection index lazily on the request object.
            detection_idx = getattr(req, "_loop_detection_idx", None)
            if detection_idx is None:
                req._loop_detection_idx = len(req.generated_ids)
            elif len(req.generated_ids) - req._loop_detection_idx >= 40:
                # Force stop: the loop has not resolved in 40 tokens.
                print(
                    f"[DiffKV] WARNING: repetition loop for session {req.session_id} "
                    f"persisted for 40 tokens after detection — forcing EOS."
                )
                is_eos = True

        if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            print(f"[DiffKV Debug] Emitting token={token_id} (is_eos={is_eos}, is_max={is_max}) generated_len={len(req.generated_ids)}")

        # Update SRL recent generated tokens to keep query routing in sync with output
        srl_state = self.wrapper.manager.get_srl_state(req.session_id)
        if srl_state is not None:
            srl_state.update_generated_tokens(token_id)
            srl_state.update_query_segment(token_id)
            curr_slot_id = srl_state.ordered_slot_ids[-1] if srl_state.ordered_slot_ids else 0
            srl_state.generated_token_slots.append(curr_slot_id)
            srl_state.update_dynamic_anchors(self.wrapper.manager._stop_token_ids)

        # Standard robust incremental decode: decodes the full generated sequence and computes the delta text.
        # This is 100% correct, handles token boundaries perfectly, and completely eliminates spacing corruption.
        all_text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
        delta_text = all_text[len(req.decoded_text):]
        req.decoded_text = all_text

        if is_eos or is_max:
            req.is_finished = True
            # ── Prefix Registry Update ─────────────────────────────────────────
            # Store the exact token IDs that were processed: prompt_ids + generated_ids.
            # These are the EXACT tokens at KV block positions 0..cached_len-1, so
            # Turn 2's prompt_ids[:cached_len] == stored_ids[:cached_len] is guaranteed.
            #
            # Why NOT retokenize the response text?
            #   The response text (e.g. "Four.") retokenizes to DIFFERENT tokens than
            #   what was generated one-by-one during decode due to BPE context effects.
            #   req.generated_ids contains the exact decode-time tokens.
            #
            # Why NOT just use kv_manager._session_token_ids (prompt-only)?
            #   That only has 42 tokens but cached_len=46 (42 prompt + 4 decode).
            #   The comparison would use compare_len=42, miss the decode tokens, and
            #   set req.cached_len=46 — which would try to skip 46 tokens including
            #   positions 42-45 that were never compared!
            prefix_ids = req.prompt_ids + req.generated_ids
            self.session_token_ids[req.session_id] = prefix_ids
            # Also update kv_manager._session_token_ids for SRL index consistency
            try:
                import torch as _torch
                _sid_dict = getattr(self.wrapper.manager, "_session_token_ids", None)
                if _sid_dict is not None:
                    _sid_dict[req.session_id] = _torch.tensor(prefix_ids, dtype=_torch.long)
            except Exception:
                pass  # Non-fatal

            # ── Reference-formatting normalisation (Fix 1) ───────────────────
            # The model sometimes generates inconsistent citation styles within
            # the same reference list (e.g. "* [1] ...", "[2] ...", "In [3] ...").
            # We normalise the completed text to a uniform "[N] ..." style.
            final_buffer_text = delta_text
            candidate = "".join(req.buffer) + final_buffer_text
            normalized = _normalize_references(candidate)
            req.chunks_queue.put_nowait({
                "text": normalized,
                "is_final": True
            })
            req.buffer.clear()
            return

        req.buffer.append(delta_text)

        # Flush immediately for smooth, token-by-token real-time streaming
        req.chunks_queue.put_nowait({
            "text": "".join(req.buffer),
            "is_final": False
        })
        req.buffer.clear()
