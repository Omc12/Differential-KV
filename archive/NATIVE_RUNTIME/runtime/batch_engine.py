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
        
        # We need a generic pad for KV caches: [B, num_heads, seq_len, head_dim]
        # For Qwen2, num_key_value_heads might be different from num_heads.
        
    def start(self):
        if not self.is_running:
            self.is_running = True
            self._loop_task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()

    async def submit(self, session_id: str, payload: Dict) -> asyncio.Queue:
        req = BatchRequest(
            session_id=session_id,
            prompt=payload["prompt"],
            max_tokens=payload.get("max_tokens", 512),
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.9),
            repetition_penalty=payload.get("repetition_penalty", 1.15)
        )
        
        encoded = self.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False)
        req.prompt_ids = encoded.input_ids[0].tolist()
        
        await self.incoming_queue.put(req)
        return req.chunks_queue

    async def _batch_loop(self):
        while self.is_running:
            # 1. Drain incoming queue
            while not self.incoming_queue.empty() and len(self.active_requests) < self.max_batch_size:
                req = await self.incoming_queue.get()
                self.active_requests.append(req)
                
            if not self.active_requests:
                # Wait for new requests
                req = await self.incoming_queue.get()
                self.active_requests.append(req)
                continue
                
            # 2. Filter cancelled requests
            self.active_requests = [r for r in self.active_requests if not r.cancelled and not r.is_finished]
            
            if not self.active_requests:
                continue
                
            try:
                await self._step()
            except Exception as e:
                print(f"Error in batch step: {e}")
                import traceback
                traceback.print_exc()
                # Fail all active requests
                for req in self.active_requests:
                    req.chunks_queue.put_nowait({"error": str(e), "is_final": True})
                    req.is_finished = True
                self.active_requests.clear()
            
            # Yield to event loop
            await asyncio.sleep(0.001)

    async def _step(self):
        step_start = time.time()
        
        # Partition into PREFILL and DECODE
        prefill_reqs = [r for r in self.active_requests if not r.is_prefilled]
        decode_reqs = [r for r in self.active_requests if r.is_prefilled]
        
        # Prefill Step (Sequential for now to establish baseline caches)
        for req in prefill_reqs:
            input_ids = torch.tensor([req.prompt_ids], device=self.wrapper.device)
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.wrapper.device).unsqueeze(0)
            
            # Differential KV Ownership: Inject session ID into model
            self.wrapper.model._diffkv_session_ids = [req.session_id]
            
            with torch.no_grad():
                out = self.wrapper.model(
                    input_ids=input_ids, 
                    position_ids=position_ids,
                    use_cache=True
                )
            
            req.is_prefilled = True
            logits = out.logits[:, -1, :]
            
            next_id = self._sample(logits, req)
            req.generated_ids.append(next_id)
            self._emit_token(req, next_id, step_start)
            
        if not decode_reqs:
            return
            
        # PADDED BATCH DECODE USING DIFFERENTIAL KV
        B = len(decode_reqs)
        input_ids = torch.tensor([[r.generated_ids[-1]] for r in decode_reqs], device=self.wrapper.device)
        
        # Compute accurate seq lengths directly from tracked request state
        # position_ids for decode = total tokens seen so far (prompt + generated - 1 for current token)
        seq_lens = [len(r.prompt_ids) + len(r.generated_ids) - 1 for r in decode_reqs]
        max_seq_len = max(seq_lens)
                
        # Construct attention_mask and position_ids
        attention_mask = torch.zeros((B, max_seq_len + 1), dtype=torch.long, device=self.wrapper.device)
        position_ids = torch.zeros((B, 1), dtype=torch.long, device=self.wrapper.device)
        
        for b_idx, slen in enumerate(seq_lens):
            attention_mask[b_idx, -(slen+1):] = 1  # attend to all history + new token
            position_ids[b_idx, 0] = slen           # position of the new token
            
        # 3. Inject session IDs for the interceptor
        self.wrapper.model._diffkv_session_ids = [r.session_id for r in decode_reqs]

        # 4. Fire Batched Decode!
        # Notice: We NO LONGER pass past_key_values! Differential KV owns the cache!
        with torch.no_grad():
            out = self.wrapper.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True
            )
            
        logits = out.logits[:, -1, :] # [B, vocab]
        
        # Sample and emit
        for b_idx, req in enumerate(decode_reqs):
            req_logits = logits[b_idx:b_idx+1]
            next_id = self._sample(req_logits, req)
            req.generated_ids.append(next_id)
            self._emit_token(req, next_id, step_start)

    def _sample(self, logits: torch.Tensor, req: BatchRequest) -> int:
        if req.repetition_penalty != 1.0:
            for tok_id in set(req.generated_ids[-64:]):
                if tok_id < logits.shape[-1]:
                    if logits[0, tok_id] > 0:
                        logits[0, tok_id] /= req.repetition_penalty
                    else:
                        logits[0, tok_id] *= req.repetition_penalty

        if req.temperature <= 0.01:
            return torch.argmax(logits, dim=-1).item()

        logits = logits / req.temperature
        probs = torch.softmax(logits, dim=-1)

        if req.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumulative - sorted_probs) > req.top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).item()
        
        return torch.multinomial(probs, num_samples=1).item()

    def _emit_token(self, req: BatchRequest, token_id: int, step_start: float):
        if req.first_token_time is None:
            req.first_token_time = time.time()
            
        if token_id == self.tokenizer.eos_token_id or len(req.generated_ids) >= req.max_tokens:
            req.is_finished = True
            if req.buffer:
                req.chunks_queue.put_nowait({
                    "text": "".join(req.buffer),
                    "is_final": True
                })
            else:
                req.chunks_queue.put_nowait({"text": "", "is_final": True})
            return
            
        text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        req.buffer.append(text)
        
        # Streaming flush logic
        FLUSH_CHARS = {'.', '!', '?', '\n', '。', '！', '？', ':', ';'}
        should_flush = len(req.buffer) >= 6 or any(c in text for c in FLUSH_CHARS)
        
        if should_flush:
            out_text = "".join(req.buffer)
            req.buffer.clear()
            req.chunks_queue.put_nowait({
                "text": out_text,
                "is_final": False
            })
