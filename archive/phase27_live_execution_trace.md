# Phase 27 — Live Execution Trace
## What ACTUALLY executes when a request hits the server

---

## Trace Method

This trace was constructed by reading the actual call chain in source files — every arrow below is a direct function call confirmed in code, not inferred from design documents.

---

## Full Hotpath Trace (Single Chat Request)

```
[OpenWebUI / curl]
  POST http://localhost:8080/v1/chat/completions
  Body: { model, messages, stream: true }

↓

[openai_compatible_api_gateway.py :: chat_completions()]
  - session_id = request.session_id or uuid4()
  - history = session_manager.get_history(session_id)     ← real message list from ProductionSessionManager
  - messages = history + [new_user_message]
  - prompt = tokenizer.apply_chat_template(messages, ...)  ← real Qwen2 chat template applied
  - payload = { prompt, max_tokens, temperature, top_p, repetition_penalty }
  - queue = await resolver.submit(session_id, payload)    ← submits to ContinuousBatchEngine
  - StreamingResponse(_stream_response(...))              ← SSE loop begins

↓

[batch_engine.py :: ContinuousBatchEngine.submit()]
  - req = BatchRequest(session_id, prompt, max_tokens, ...)
  - encoded = tokenizer(req.prompt, ...)                  ← real tokenizer call
  - req.prompt_ids = encoded.input_ids[0].tolist()
  - await incoming_queue.put(req)
  - returns req.chunks_queue (asyncio.Queue)

↓

[batch_engine.py :: _batch_loop() → _step()]
  - Partitions requests: prefill_reqs / decode_reqs

─── PREFILL STEP ────────────────────────────────────────

  input_ids = torch.tensor([req.prompt_ids], device=cuda)
  position_ids = torch.arange(0, seq_len)
  model._dkv_session_ids = [req.session_id]           ← session injection

  model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
  ↓
  [DKV patched forward in every Qwen2 layer]
  (runtime/dkv_attention.py :: dkv_forward())

    is_decode = (use_cache and q_len == 1 and bsz == 1)  → FALSE for prefill

    query_states = self.q_proj(hidden_states)             ← CUDA kernel: linear
    key_states   = self.k_proj(hidden_states)             ← CUDA kernel: linear
    value_states = self.v_proj(hidden_states)             ← CUDA kernel: linear

    apply_rotary_pos_emb(query_states, key_states, cos, sin)  ← CUDA kernel: RoPE

    kv_manager.ingest_streaming(sid, layer_idx, k, v)
    ↓
    [StreamingSparseIngestManager.ingest_chunk()]
      - Streams tokens through micro_block_size=16 windows
      - Extracts anchor token (1 dense token per block)
      - Accumulates into StreamingKVBlock.active_k / active_v
      - When block fills: _submit_block_for_compression()
        ↓
        [AsyncCompressor.submit()]
          - k.detach().to("cpu", non_blocking=True)       ← async D2H transfer
          - queue.put_nowait((block, k_cpu, v_cpu))        ← background queue

          [Background worker thread:]
            k_gpu = k_cpu.to(block.anchor_kv.device)     ← H2D transfer
            _compress_block_sync(block, k_gpu, v_gpu)
            ↓
            AdaptiveRankSelector.select_rank(deltas)      ← selects rank 4/8/16/32
            compress_lowrank(deltas, rank)
            ↓
            torch.linalg.svd(deltas)                      ← CUDA/CPU SVD
            block.U, block.V, block.scale = lr_delta
            block.active_k = None                         ← VRAM freed

    F.scaled_dot_product_attention(query, key, value,
        attn_mask=None, is_causal=(q_len > 1))             ← CUDA: Flash/SDPA kernel

    [LM head patch]
    lm_head(hidden_states[:, -1:, :])                      ← last-token only projection
    → logits [1, 1, vocab_size]

  req.is_prefilled = True
  logits = out.logits[:, -1, :]
  next_id = _sample(logits, req)                           ← temperature/top-p/rep-pen
  _emit_token(req, next_id)

─── DECODE STEP ─────────────────────────────────────────

  input_ids = [[last_generated_token_id]]                  ← shape [B, 1]
  model._dkv_session_ids = [sid for each decode req]

  model(input_ids, attention_mask, position_ids, use_cache=True)
  ↓
  [dkv_forward() — every layer]

    is_decode = True (q_len == 1, bsz can be > 1)

    query/key/value projections                            ← CUDA: 3 linear kernels
    apply_rotary_pos_emb                                   ← CUDA: RoPE

    kv_manager.ingest_streaming(sid, layer_idx, curr_k, curr_v)
    → StreamingSparseIngestManager.append_decode_token()
      → ingest_chunk() with T=1 token

    blocks = kv_manager.get_streaming_blocks(sid, layer_idx)

    last_block = blocks[-1]
    active_k = last_block.active_k    ← current dense window
    history_blocks = blocks[:-1]

    sparse_batch = build_sparse_batch(history_blocks)
    ↓
    [batched_sparse_attn.py :: build_sparse_batch()]
      - Stacks compressed blocks: anchors_K, anchors_V, V_K, V_V, U
      - Returns SparseBatch(N, S_max, R)

    batched_sparse_attn_decode(q, sparse_batch, dense_history, active_k, active_v)
    ↓
    [batched_sparse_attn.py :: batched_sparse_attn_decode()]
      GPU OP 1: s_anchor = (q * aK).sum(-1)              ← CUDA: elementwise + reduction
      GPU OP 2: delta_scores = einsum("nhr,nsr->nhs",     ← CUDA: batched matmul
                    q_proj, batch.U) * scales
      GPU OP 3: VV_t = VV.permute(0, 2, 1, 3)            ← CUDA: transpose
      Python loop over N blocks: FlashAttention accumulators (scalar arithmetic, no kernel launches)
      _update_dense_hq(active_k, active_v)                ← CUDA: dense window attention

    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    self.o_proj(attn_output)                              ← CUDA: linear

  logits = out.logits[:, -1, :]                           ← last-token (lm_head patched)
  for each request: _sample(logits[b_idx]) → next_id
  _emit_token() → chunks_queue.put_nowait({text, is_final})

↓

[gateway._stream_response()]
  chunk = await queue.get()
  yield f"data: {json.dumps(data)}\n\n"                   ← SSE chunk to OpenWebUI

↓

[session_manager.append_message(session_id, "assistant", full_text)]
```

---

## Kernel Dispatch Inventory (Per Decode Step)

| Kernel | Source | Dispatch Type |
|---|---|---|
| q_proj, k_proj, v_proj linear | PyTorch aten | Real CUDA via cuBLAS/cuDNN |
| apply_rotary_pos_emb | PyTorch aten | Real CUDA elementwise |
| StreamingKVBlock accumulation | Python list ops | CPU — no kernel |
| AsyncCompressor SVD | torch.linalg.svd | Real CUDA or CPU eigendecomp |
| build_sparse_batch: torch.stack, torch.cat | PyTorch aten | Real CUDA if tensors on GPU |
| batched_sparse_attn einsum Op 1 | PyTorch aten::einsum | Real CUDA matmul |
| batched_sparse_attn einsum Op 2 | PyTorch aten::einsum | Real CUDA batched matmul |
| batched_sparse_attn dense window | PyTorch aten ops | Real CUDA |
| o_proj linear | PyTorch aten | Real CUDA via cuBLAS |
| F.scaled_dot_product_attention (prefill) | PyTorch SDPA | Real Flash/SDPA CUDA kernel |
| lm_head linear (last token) | PyTorch aten | Real CUDA via cuBLAS |
| TritonDKV.reconstruct_lowrank | Triton JIT | Real Triton GPU kernel (with PyTorch fallback) |

---

## Kernels That Do NOT Dispatch (Despite Code Existing)

| Kernel | Reason |
|---|---|
| _fused_sparse_decode_kernel (Triton, triton_sparse_attn.py) | Requires NativeBlockPool; dkv_core.so never compiled |
| DKVPagingStream CUDA stream transfers | paging_stream.cu never compiled |
| StaticSparseDecodeGraph graph.replay() | Never instantiated in serving path |
| NCCL all_reduce / all_gather | Code commented out in stubs; never called |
| Any dist.* call | torch.distributed never initialized |

---

## Sparse Ratio in Practice

At decode time, every layer processes:
- **N compressed blocks** (U, V, anchor): O(block×rank) VRAM instead of O(block×seq×dim)
- **1 active dense block** (micro_block_size=16 tokens max): irreducible dense window

Effective dense footprint per layer per session:
`= 1 anchor (irreducible) + up to 16 active tokens × kv_heads × head_dim × 2 × fp16_bytes`

All older blocks are compressed via SVD — their memory is rank×seq vs heads×seq×head_dim.
