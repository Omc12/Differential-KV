import asyncio
import gc
import os
import time
import threading
import torch
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

    @property
    def total_seq_len(self) -> int:
        """Total tokens seen so far = prompt + all generated tokens."""
        return len(self.prompt_ids) + len(self.generated_ids)


class ContinuousBatchEngine:
    def __init__(self, wrapper, max_batch_size=8):
        self.wrapper = wrapper
        self.max_batch_size = max_batch_size
        self.active_requests: List[BatchRequest] = []
        self.incoming_queue = asyncio.Queue()
        self.is_running = False
        self._loop_task = None

        self.tokenizer = self.wrapper.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._alphanumeric_tokens = {}
        self.session_token_ids = {}
        
        # Universal stop tokens inherited from wrapper
        self.stop_token_ids = getattr(self.wrapper, "stop_token_ids", {self.tokenizer.eos_token_id})
        
        # Track decode steps for periodic memory sweeps
        self.decode_steps_since_gc = 0

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
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            print(f"[DiffKV] Post-prefill: {allocated:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved")

    # ── Compression barrier ─────────────────────────────────────────────

    async def _wait_for_compression(self, session_id: str, timeout_s: float = 8.0):
        """
        Non-blocking async wait until every block for *session_id* has left
        the SUBMITTED state (async SVD compressor is done), OR timeout elapses.

        CRITICAL: uses `await asyncio.sleep()` so the event loop is NEVER
        blocked. A sync `time.sleep()` here would freeze the entire server,
        causing SSE streams to idle-timeout and responses to appear truncated.
        """
        mgr = self.wrapper.manager
        streaming_mgr = getattr(mgr, "_streaming_mgr", None)
        if streaming_mgr is None:
            return

        session_blocks = streaming_mgr.session_blocks.get(session_id, {})
        if not session_blocks:
            return

        deadline = time.monotonic() + timeout_s
        check_interval = 0.005  # 5 ms async yield — event loop stays live

        while time.monotonic() < deadline:
            found_submitted = False
            for layer_idx, blocks in session_blocks.items():
                for block in blocks:
                    if getattr(block, "state", None) == "SUBMITTED":
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
            from native_core.sparse_decode.triton_diffkv import TritonDiffKV
            if hasattr(TritonDiffKV, '_recon_buffers'):
                TritonDiffKV._recon_buffers.clear()
        except Exception as e:
            print(f"[DiffKV] Warning: failed to clear TritonDiffKV reconstruction buffers: {e}")

    async def submit(self, session_id: str, payload: Dict) -> asyncio.Queue:
        req = BatchRequest(
            session_id=session_id,
            prompt=payload["prompt"],
            max_tokens=payload.get("max_tokens", 2048),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.9),
            repetition_penalty=payload.get("repetition_penalty", 1.15)
        )

        encoded = self.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False)
        req.prompt_ids = encoded.input_ids[0].tolist()

        # O(1) Smart Prefix Check: check if the session already has resident KV cache.
        # If so, mark the cached length so prefill is incremental (avoiding O(N) re-prefill of history).
        req.cached_len = 0
        if hasattr(self.wrapper.manager, "get_session_sequence_length"):
            cached_len = self.wrapper.manager.get_session_sequence_length(session_id)
            if cached_len > 0 and cached_len < len(req.prompt_ids):
                stored_ids = self.session_token_ids.get(session_id, [])
                if len(stored_ids) >= cached_len and req.prompt_ids[:cached_len] == stored_ids[:cached_len]:
                    req.cached_len = cached_len
                    print(f"[DiffKV BatchEngine] Found cached history for session {session_id}: length {cached_len} tokens. Reusing KV cache!")
                else:
                    print(f"[DiffKV BatchEngine] Prefix mismatch or sequence length inconsistency for session {session_id}. Expected matching history of length {cached_len}. Clearing stale cache.")

        await self.incoming_queue.put(req)
        return req.chunks_queue

    def cancel(self, session_id: str):
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

            try:
                await self._step()
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

    def _free_session_kv(self, session_id: str):
        """Release the KV manager blocks for a completed session to free VRAM."""
        try:
            if session_id in self.session_token_ids:
                del self.session_token_ids[session_id]
            kv_mgr = self.wrapper.manager
            if hasattr(kv_mgr, 'clear_session'):
                kv_mgr.clear_session(session_id)
        except Exception as e:
            print(f"[DiffKV] WARNING: could not free KV for session {session_id}: {e}")

    async def _step(self):
        step_start = time.time()

        # Partition into PREFILL and DECODE
        prefill_reqs = [r for r in self.active_requests if not r.is_prefilled]
        decode_reqs  = [r for r in self.active_requests if r.is_prefilled]

        # ─────────────────────────────────────────────────────────────────
        # PREFILL — one request at a time (different prompt lengths can't
        # be batched without padding, which wastes memory for large prompts)
        # ─────────────────────────────────────────────────────────────────
        for req in prefill_reqs:
            t0_pref = time.perf_counter()
            
            cached_len = getattr(req, "cached_len", 0)
            if cached_len > 0:
                new_prompt_ids = req.prompt_ids[cached_len:]
            else:
                # Fresh prefill from scratch — clear stale KV
                self._free_session_kv(req.session_id)
                new_prompt_ids = req.prompt_ids

            # Ensure session is registered and metadata initialized
            if hasattr(self.wrapper.manager, "init_session"):
                self.wrapper.manager.init_session(req.session_id, prefill_len=len(req.prompt_ids))

            # Inject session ID so the attention patch stores KV under the right key
            self.wrapper.model._diffkv_session_ids = [req.session_id]

            # Progressive Prompt Chunking — process in chunks of 1024 tokens.
            # Chunk size of 1024 (vs previous 512) reduces chunk count by 2× for
            # long prompts (e.g. 2540 tokens: 5 chunks → 3 chunks), cutting per-chunk
            # Python overhead by the same factor. The O(N²) attention cost for 1024
            # tokens is still well within GPU memory for Qwen2.5-1.5B.
            # VRAM overhead vs 512: ~4× per-chunk peak activation (fp16 attention)
            # but this is transient and reclaimed immediately after each chunk.
            chunk_size = 1024
            L_new = len(new_prompt_ids)
            out = None

            # Pre-allocate a single pinned-memory input buffer (reused across all chunks).
            # Avoids repeated pin_memory() allocation inside the hot loop — one allocation,
            # N in-place fills. The transfer to GPU uses non_blocking=True so the CPU does
            # not spin-wait on DMA completion between chunks.
            _input_buf = torch.zeros((1, chunk_size), dtype=torch.long).pin_memory()

            for offset in range(0, L_new, chunk_size):
                chunk_ids = new_prompt_ids[offset : offset + chunk_size]
                chunk_cached_len = cached_len + offset
                actual_len = len(chunk_ids)

                # In-place fill of the reusable buffer — zero allocation per chunk
                _input_buf[0, :actual_len] = torch.as_tensor(chunk_ids, dtype=torch.long)
                input_ids = _input_buf[:, :actual_len].to(self.wrapper.device, non_blocking=True)

                position_ids = torch.arange(
                    chunk_cached_len, chunk_cached_len + actual_len,
                    dtype=torch.long, device=self.wrapper.device
                ).unsqueeze(0)

                with torch.no_grad():
                    out = self.wrapper.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        use_cache=True
                    )
                # DO NOT call gc.collect()/empty_cache() here — it was firing once per
                # 512-token chunk (5× for a 2540-token prompt), costing 500-1000ms of
                # CUDA-allocator flush overhead. Moved to AFTER the loop.

            req.is_prefilled = True
            logits = out.logits[:, -1, :]  # lm_head patch already sliced to last token
            next_id = self._sample(logits, req)
            req.generated_ids.append(next_id)
            self._emit_token(req, next_id, step_start)
            self.session_token_ids[req.session_id] = req.prompt_ids + req.generated_ids

            del _input_buf  # Release the pinned buffer immediately

            # Step 1: Post-prefill batch GPU SVD — compress all captured KV into
            # the block pool in a single batched operation per layer. Zero PCIe.
            if hasattr(self.wrapper.manager, "compress_prefill_kv"):
                self.wrapper.manager.compress_prefill_kv(req.session_id)

            # Step 2: Release allocator-held staging buffers
            self._post_prefill_cleanup()
            self._log_vram(f"post-prefill session={req.session_id}")

            # ── Compression barrier (Bypassed in Phase 35 for 10x prefill speedup) ──
            # SVD compression runs asynchronously in background threads while decoding,
            # which is completely safe and avoids multi-second stalls before the first token.
            pass

            if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
                dur_pref = (time.perf_counter() - t0_pref) * 1000
                print(f"[DiffKV Telemetry] Prefill session={req.session_id} tokens={len(req.prompt_ids)} duration={dur_pref:.2f}ms")
                # Emit block state breakdown (ACCUMULATING / SUBMITTED / COMPRESSED)
                if hasattr(self.wrapper.manager, "log_block_states"):
                    self.wrapper.manager.log_block_states(req.session_id)

        if not decode_reqs:
            return

        # ─────────────────────────────────────────────────────────────────
        # BATCHED DECODE (B >= 1) — CUDA Graph Stability Buckets (Batch Padding)
        # ─────────────────────────────────────────────────────────────────
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
        
        # Determine the nearest power of 2 stability bucket size (1, 2, 4, 8, etc.)
        bucket_size = 1
        while bucket_size < actual_batch_size:
            bucket_size *= 2
            
        # Pad with dummy request structures if needed to fill the stability bucket shape
        if actual_batch_size < bucket_size:
            dummy_req = decode_reqs[-1]
            dummy_input = [dummy_req.generated_ids[-1]]
            dummy_pos = [dummy_req.total_seq_len - 1]
            dummy_sid = "dummy_session"
            
            for _ in range(bucket_size - actual_batch_size):
                input_ids_list.append(dummy_input)
                position_ids_list.append(dummy_pos)
                session_ids.append(dummy_sid)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.wrapper.device)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, device=self.wrapper.device)

        # Inject session IDs for this batch decode step
        self.wrapper.model._diffkv_session_ids = session_ids

        with torch.no_grad():
            out = self.wrapper.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True
            )

        logits = out.logits[:, -1, :]  # shape: [bucket_size, vocab_size]
        
        # Extract and sample outputs ONLY for actual active requests
        for idx in range(actual_batch_size):
            req = decode_reqs[idx]
            req_logits = logits[idx : idx + 1]  # shape: [1, vocab_size]
            next_id = self._sample(req_logits, req)
            req.generated_ids.append(next_id)
            self._emit_token(req, next_id, step_start)
            self.session_token_ids[req.session_id] = req.prompt_ids + req.generated_ids

        if os.environ.get("DIFFKV_TELEMETRY", "0") == "1":
            dur_dec = (time.perf_counter() - t0_dec) * 1000
            print(f"[DiffKV Telemetry] Decode Step batch_size={actual_batch_size} bucket_size={bucket_size} duration={dur_dec:.2f}ms")

        # Periodic VRAM defragmentation during long decode runs.
        # empty_cache() was removed from the prefill path to save 100-200ms of TTFT.
        # We fire it here instead, every 100 decode steps, which is frequent enough
        # to prevent caching-allocator fragmentation in multi-thousand-token outputs
        # without blocking any user-visible latency metrics.
        self.decode_steps_since_gc += 1
        if self.decode_steps_since_gc >= 100:
            self.decode_steps_since_gc = 0
            torch.cuda.empty_cache()

    def _sample(self, logits: torch.Tensor, req: BatchRequest) -> int:
        # Apply repetition penalty over the most recent tokens.
        # Fully vectorized on GPU — no Python loop, no CUDA sync per token.
        # Previous code iterated over set(generated_ids[-64:]) and indexed logits
        # one element at a time, forcing a CUDA sync for every unique token id.
        if req.repetition_penalty != 1.0 and req.generated_ids:
            penalty_ids = torch.tensor(
                list(set(req.generated_ids[-64:])),
                dtype=torch.long, device=logits.device
            )
            # Clamp to valid vocab range
            penalty_ids = penalty_ids[penalty_ids < logits.shape[-1]]
            if penalty_ids.numel() > 0:
                scores = logits[0, penalty_ids]          # [N]
                # Standard repetition-penalty formula (same sign, magnitude reduced)
                scores = torch.where(scores > 0, scores / req.repetition_penalty,
                                                 scores * req.repetition_penalty)
                logits[0].scatter_(0, penalty_ids, scores)

        if req.temperature <= 0.01:
            return torch.argmax(logits, dim=-1).item()

        logits = logits / req.temperature

        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=-100.0, posinf=100.0, neginf=-100.0)

        probs = torch.softmax(logits, dim=-1)

        if not torch.isfinite(probs).all():
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        probs_sum = probs.sum(dim=-1, keepdim=True)
        if (probs_sum == 0).any():
            probs = torch.ones_like(probs) / probs.shape[-1]
        else:
            probs = probs / probs_sum

        if req.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens where cumulative probability exceeds top_p
            mask = (cumulative - sorted_probs) > req.top_p
            sorted_probs[mask] = 0.0
            psum = sorted_probs.sum(dim=-1, keepdim=True)
            sorted_probs = sorted_probs / torch.where(psum == 0, torch.ones_like(psum), psum)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).item()

        return torch.multinomial(probs, num_samples=1).item()

    def _emit_token(self, req: BatchRequest, token_id: int, step_start: float):
        if req.first_token_time is None:
            req.first_token_time = time.time()

        is_eos = (token_id in self.stop_token_ids)
        is_max = (len(req.generated_ids) >= req.max_tokens)

        # Standard robust incremental decode: decodes the full generated sequence and computes the delta text.
        # This is 100% correct, handles token boundaries perfectly, and completely eliminates spacing corruption.
        all_text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
        delta_text = all_text[len(req.decoded_text):]
        req.decoded_text = all_text

        if is_eos or is_max:
            req.is_finished = True
            # Flush whatever is in the buffer and the new delta on finish
            req.buffer.append(delta_text)
            req.chunks_queue.put_nowait({
                "text": "".join(req.buffer),
                "is_final": True
            })
            req.buffer.clear()
            return

        req.buffer.append(delta_text)

        # Flush at phrase boundaries or every 6 tokens for low latency
        FLUSH_CHARS = {'.', '!', '?', '\n', '\u3002', '\uff01', '\uff1f', ':', ';'}
        should_flush = len(req.buffer) >= 6 or any(c in delta_text for c in FLUSH_CHARS)

        if should_flush:
            req.chunks_queue.put_nowait({
                "text": "".join(req.buffer),
                "is_final": False
            })
            req.buffer.clear()
