import logging
import time
import torch
import asyncio
from typing import Dict, Any, List, AsyncGenerator, Optional


class LGSResolver:
    """
    Clean serving resolver: HF model -> sampler -> streamer.
    No stub engines. No synthetic metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("LGSResolver")
        self.wrapper = None

    def setup_runtime(self):
        """Initialize the HF model wrapper. Idempotent."""
        if self.wrapper is not None:
            return
        from runtime.hf_dkv_wrapper import DKVHFWrapper
        model_id = self.config.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
        self.wrapper = DKVHFWrapper(
            model_id,
            {"mode": "fp16", "block_size": 64, "rank": 16,
             "prefill_chunk_size": self.config.get("prefill_chunk_size", 512)},
        )

    # ------------------------------------------------------------------
    # Core sampler — used by both streaming and non-streaming paths
    # ------------------------------------------------------------------

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,          # [batch, vocab]
        temperature: float,
        top_p: float,
        generated_ids_per_seq: List[List[int]],
        repetition_penalty: float,
    ) -> torch.Tensor:                  # returns [batch, 1]
        """
        Real sampler: repetition penalty → temperature → top-p → multinomial.
        Falls back to argmax only when temperature <= 0.01.
        """
        # 1. Repetition penalty
        if repetition_penalty != 1.0:
            for b, recent in enumerate(generated_ids_per_seq):
                for tok_id in set(recent[-64:]):
                    if tok_id < logits.shape[-1]:
                        if logits[b, tok_id] > 0:
                            logits[b, tok_id] /= repetition_penalty
                        else:
                            logits[b, tok_id] *= repetition_penalty

        # 2. Greedy shortcut
        if temperature <= 0.01:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # 3. Temperature scaling
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # 4. Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # mask tokens whose cumulative prob exceeds top_p
            # shift right by 1 so the token that crosses the threshold is kept
            mask = (cumulative - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            next_tokens = sorted_indices.gather(-1, sampled)
        else:
            next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens  # [batch, 1]

    # ------------------------------------------------------------------
    # Shared decode loop helper
    # ------------------------------------------------------------------

    def _build_prompts(
        self,
        session_ids: List[str],
        payloads: List[Dict],
        session_manager,
    ) -> List[str]:
        """
        Builds final prompt strings by merging server-side history with
        the client's current message(s).
        """
        prompts = []
        for sid, payload in zip(session_ids, payloads):
            messages = list(payload.get("messages", []))
            if session_manager is not None:
                history = session_manager.get_history(sid)
                # Only prepend server history when client sends just the latest turn
                if history and len(messages) == 1 and messages[0]["role"] == "user":
                    messages = history + messages
            prompt = self.wrapper.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts

    def _store_assistant_reply(
        self,
        session_ids: List[str],
        payloads: List[Dict],
        generated_texts: List[str],
        session_manager,
    ):
        if session_manager is None:
            return
        for sid, payload, reply in zip(session_ids, payloads, generated_texts):
            messages = payload.get("messages", [])
            history = session_manager.get_history(sid)
            if not history:
                # First turn: store entire conversation
                for msg in messages:
                    session_manager.append_message(sid, msg["role"], msg["content"])
            session_manager.append_message(sid, "assistant", reply)

    # ------------------------------------------------------------------
    # Streaming executor
    # ------------------------------------------------------------------

    async def lgs_runtime_stream_executor(
        self,
        session_ids: List[str],
        payloads: List[Dict],
        session_manager=None,
    ) -> AsyncGenerator[Dict, None]:
        """
        True autoregressive streaming with:
        - temperature + top-p sampling
        - repetition penalty
        - phrase-group chunk buffering (no word-by-word rendering)
        - server-side conversation history
        """
        self.setup_runtime()
        request_ts = time.time()

        # Generation params (from first payload; applied to all in batch)
        temperature = max(0.01, min(2.0, float(payloads[0].get("temperature", 0.7))))
        top_p       = max(0.05, min(1.0,  float(payloads[0].get("top_p",       0.9))))
        rep_penalty = max(1.0,  min(2.0,  float(payloads[0].get("repetition_penalty", 1.15))))
        max_gen     = max(p.get("max_tokens", 512) for p in payloads)

        # Build prompts (merges server-side history)
        prompts = self._build_prompts(session_ids, payloads, session_manager)

        tok = self.wrapper.tokenizer
        tok.pad_token = tok.eos_token
        encoded = tok(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=8192,
        ).to(self.wrapper.device)
        input_ids = encoded.input_ids         # [B, L]
        prompt_len = input_ids.shape[1]

        # Prefill
        decode_start_ts = time.time()
        for i, sid in enumerate(session_ids):
            self.wrapper.forward_step(input_ids[i:i+1], session_id=sid)

        eos_id    = tok.eos_token_id
        pad_id    = tok.pad_token_id or eos_id
        finished  = [False] * len(session_ids)
        cur_ids   = input_ids
        gen_ids   = [[] for _ in session_ids]    # for repetition penalty
        buffers   = [[] for _ in session_ids]    # text chunk buffers
        all_gen   = [[] for _ in session_ids]    # full generated token ids

        CHUNK_TOKENS  = 6
        FLUSH_CHARS   = {'.', '!', '?', '\n', '。', '！', '？', ':', ';'}
        first_tok_ts  = None

        for step in range(max_gen):
            if all(finished):
                break

            step_ts = time.time()
            if first_tok_ts is None:
                first_tok_ts = step_ts

            # One forward pass per active session (incremental: only last token)
            logits_list = []
            vocab_size = self.wrapper.model.config.vocab_size
            for i, sid in enumerate(session_ids):
                if not finished[i]:
                    logit = self.wrapper.forward_step(
                        cur_ids[i:i+1, -1:], session_id=sid
                    )
                    logits_list.append(logit)
                else:
                    logits_list.append(
                        torch.zeros(1, vocab_size, device=self.wrapper.device)
                    )

            logits = torch.cat(logits_list, dim=0)  # [B, vocab]

            # Sample with real sampler
            next_toks = self._sample_next_tokens(
                logits, temperature, top_p, gen_ids, rep_penalty
            )  # [B, 1]

            # Mask finished sequences
            for i in range(len(session_ids)):
                if finished[i]:
                    next_toks[i, 0] = pad_id

            cur_ids = torch.cat([cur_ids, next_toks], dim=-1)

            # Collect chunks to emit this step
            decode_ts = time.time()
            chunks_out = []

            for i, sid in enumerate(session_ids):
                if finished[i]:
                    continue

                tok_id = next_toks[i, 0].item()
                gen_ids[i].append(tok_id)
                all_gen[i].append(tok_id)

                if tok_id == eos_id:
                    finished[i] = True
                    # Flush remainder
                    if buffers[i]:
                        chunks_out.append({
                            "session_id": sid,
                            "text": "".join(buffers[i]),
                            "is_final": True,
                            "decode_ts": decode_ts,
                        })
                        buffers[i].clear()
                    continue

                tok_text = tok.decode([tok_id], skip_special_tokens=True)
                buffers[i].append(tok_text)

                should_flush = (
                    len(buffers[i]) >= CHUNK_TOKENS
                    or any(c in tok_text for c in FLUSH_CHARS)
                )
                if should_flush:
                    text = "".join(buffers[i])
                    buffers[i].clear()
                    if text:
                        chunks_out.append({
                            "session_id": sid,
                            "text": text,
                            "is_final": False,
                            "decode_ts": decode_ts,
                        })

            if chunks_out:
                yield {
                    "step": step,
                    "chunks": chunks_out,
                    "server_timings": {
                        "request_ts": request_ts,
                        "first_tok_ts": first_tok_ts,
                        "step_ms": (decode_ts - step_ts) * 1000,
                    },
                }

            if step % 4 == 0:
                await asyncio.sleep(0)

        # Final buffer flush
        final_chunks = []
        for i, sid in enumerate(session_ids):
            if buffers[i]:
                final_chunks.append({
                    "session_id": sid,
                    "text": "".join(buffers[i]),
                    "is_final": True,
                    "decode_ts": time.time(),
                })
        if final_chunks:
            yield {"step": -1, "chunks": final_chunks, "server_timings": {}}

        # Persist assistant replies into session history
        generated_texts = [
            tok.decode(all_gen[i], skip_special_tokens=True)
            for i in range(len(session_ids))
        ]
        self._store_assistant_reply(session_ids, payloads, generated_texts, session_manager)

        yield {
            "is_done": True,
            "server_timings": {
                "request_ts": request_ts,
                "first_tok_ts": first_tok_ts,
                "total_ms": (time.time() - decode_start_ts) * 1000,
            },
        }

    # ------------------------------------------------------------------
    # Non-streaming executor (same sampler)
    # ------------------------------------------------------------------

    async def lgs_runtime_executor(
        self,
        session_ids: List[str],
        payloads: List[Dict],
        session_manager=None,
    ) -> List[Dict]:
        """Generate-then-return path. Identical sampler to streaming."""
        self.setup_runtime()
        request_ts = time.time()

        temperature = max(0.01, min(2.0, float(payloads[0].get("temperature", 0.7))))
        top_p       = max(0.05, min(1.0,  float(payloads[0].get("top_p",       0.9))))
        rep_penalty = max(1.0,  min(2.0,  float(payloads[0].get("repetition_penalty", 1.15))))
        max_gen     = max(p.get("max_tokens", 512) for p in payloads)

        prompts = self._build_prompts(session_ids, payloads, session_manager)

        tok = self.wrapper.tokenizer
        tok.pad_token = tok.eos_token
        encoded = tok(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=8192,
        ).to(self.wrapper.device)
        input_ids = encoded.input_ids
        prompt_len = input_ids.shape[1]

        decode_start_ts = time.time()
        for i, sid in enumerate(session_ids):
            self.wrapper.forward_step(input_ids[i:i+1], session_id=sid)

        eos_id   = tok.eos_token_id
        pad_id   = tok.pad_token_id or eos_id
        finished = [False] * len(session_ids)
        cur_ids  = input_ids
        gen_ids  = [[] for _ in session_ids]

        for step in range(max_gen):
            if all(finished):
                break

            vocab_size = self.wrapper.model.config.vocab_size
            logits_list = []
            for i, sid in enumerate(session_ids):
                if not finished[i]:
                    logit = self.wrapper.forward_step(
                        cur_ids[i:i+1, -1:], session_id=sid
                    )
                    logits_list.append(logit)
                else:
                    logits_list.append(
                        torch.zeros(1, vocab_size, device=self.wrapper.device)
                    )

            logits = torch.cat(logits_list, dim=0)
            next_toks = self._sample_next_tokens(
                logits, temperature, top_p, gen_ids, rep_penalty
            )

            for i in range(len(session_ids)):
                if finished[i]:
                    next_toks[i, 0] = pad_id

            cur_ids = torch.cat([cur_ids, next_toks], dim=-1)

            for i in range(len(session_ids)):
                tok_id = next_toks[i, 0].item()
                gen_ids[i].append(tok_id)
                if tok_id == eos_id:
                    finished[i] = True

            if step % 5 == 0:
                await asyncio.sleep(0)

        end_ts = time.time()
        results = []
        generated_texts = []
        for i, sid in enumerate(session_ids):
            gen_token_ids = cur_ids[i, prompt_len:]
            text = tok.decode(gen_token_ids, skip_special_tokens=True)
            generated_texts.append(text)
            results.append({
                "text": text,
                "prompt_tokens": prompt_len,
                "completion_tokens": len(gen_ids[i]),
                "total_tokens": prompt_len + len(gen_ids[i]),
                "server_timings": {
                    "request_ts": request_ts,
                    "decode_start_ts": decode_start_ts,
                    "end_ts": end_ts,
                    "total_ms": (end_ts - decode_start_ts) * 1000,
                },
            })

        self._store_assistant_reply(session_ids, payloads, generated_texts, session_manager)
        return results
