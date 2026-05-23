import asyncio
import time
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
            max_tokens=payload.get("max_tokens", 512),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.9),
            repetition_penalty=payload.get("repetition_penalty", 1.15)
        )

        # Tokenize WITHOUT stripping special tokens — the chat template already
        # added them as plain text characters, so add_special_tokens=False is
        # correct here (avoids double-adding BOS etc.).
        encoded = self.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False)
        req.prompt_ids = encoded.input_ids[0].tolist()

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

            # 2. Filter out finished/cancelled requests, freeing their KV cache
            for req in self.active_requests:
                if req.is_finished or req.cancelled:
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
            # Always clear any stale KV for this session before prefill
            self._free_session_kv(req.session_id)

            input_ids    = torch.tensor([req.prompt_ids], dtype=torch.long).pin_memory().to(self.wrapper.device, non_blocking=True)
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long,
                                        device=self.wrapper.device).unsqueeze(0)

            # Inject session ID so the attention patch stores KV under the right key
            self.wrapper.model._diffkv_session_ids = [req.session_id]

            with torch.no_grad():
                out = self.wrapper.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True
                )

            req.is_prefilled = True
            logits = out.logits[:, -1, :]  # lm_head patch already sliced to last token
            next_id = self._sample(logits, req)
            req.generated_ids.append(next_id)
            self._emit_token(req, next_id, step_start)

        if not decode_reqs:
            return

        # ─────────────────────────────────────────────────────────────────
        # BATCHED DECODE (B >= 1) — extremely fast parallel execution
        # ─────────────────────────────────────────────────────────────────
        input_ids_list = []
        position_ids_list = []
        session_ids = []

        for req in decode_reqs:
            cur_pos = req.total_seq_len - 1
            input_ids_list.append([req.generated_ids[-1]])
            position_ids_list.append([cur_pos])
            session_ids.append(req.session_id)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long).pin_memory().to(self.wrapper.device, non_blocking=True)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long).pin_memory().to(self.wrapper.device, non_blocking=True)

        # Inject session IDs for this batch decode step
        self.wrapper.model._diffkv_session_ids = session_ids

        with torch.no_grad():
            out = self.wrapper.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True
            )

        logits = out.logits[:, -1, :]  # shape: [B, vocab_size]
        for idx, req in enumerate(decode_reqs):
            req_logits = logits[idx : idx + 1]  # shape: [1, vocab_size]
            next_id = self._sample(req_logits, req)
            req.generated_ids.append(next_id)
            self._emit_token(req, next_id, step_start)

    def _sample(self, logits: torch.Tensor, req: BatchRequest) -> int:
        # Apply repetition penalty over the most recent tokens
        if req.repetition_penalty != 1.0:
            for tok_id in set(req.generated_ids[-64:]):
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
                        logits[0, tok_id] /= req.repetition_penalty
                    else:
                        logits[0, tok_id] *= req.repetition_penalty

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

        is_eos = (token_id == self.tokenizer.eos_token_id)
        is_max = (len(req.generated_ids) >= req.max_tokens)

        # Sequence-delta decoding to resolve multi-byte characters and spacing correctly
        new_text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
        delta_text = new_text[len(req.decoded_text):]
        req.decoded_text = new_text

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
