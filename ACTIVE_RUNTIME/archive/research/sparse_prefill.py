"""
runtime/sparse_prefill.py

Phase 13 — Sparse Prefill Execution

Reduces the catastrophic O(N^2) memory and compute cost of long-context prefill
by replacing full dense prefill with Chunked Sparse Prefill.

Mechanism:
Instead of a single massive Q x K attention over 8192+ tokens, we slice the
prefill sequence into chunks.
For each chunk, Q attends ONLY to:
1. Sinks (first N tokens, usually 4-64)
2. Local Context (the previous chunk)
3. Causal Self (the current chunk)

FLOP Reduction:
Dense Prefill: O(N^2)
Sparse Chunked: O(N * (Sink + Local + Chunk_Size)) = O(N) linear time!

Memory Movement:
We avoid materializing the massive N x N attention matrix. 
We only load the specific KV slices into SRAM for SDPA.

KV Materialization Reduction (Compress-on-Write):
Because a chunk only attends to the PREVIOUS chunk, any chunks older than
(current - 1) are physically completely evicted from the hot attention path.
They can be immediately passed to the async compressor without waiting for
the end of the prefill!
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import time

class SparsePrefillEngine:
    """
    Executes a sparse prefill forward pass by chunking the sequence.
    """
    def __init__(
        self,
        sink_tokens: int = 64,
        chunk_size: int = 512,
        local_window_chunks: int = 1
    ):
        self.sink_tokens = sink_tokens
        self.chunk_size = chunk_size
        self.local_window_chunks = local_window_chunks
        
        self.stats = {
            "dense_flops": 0,
            "sparse_flops": 0,
            "time_ms": 0.0
        }

    def _create_chunk_mask(self, q_len: int, k_sink: int, k_local: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Creates the attention mask for a chunk.
        Q shape: [q_len]
        K shape: [k_sink + k_local + q_len]
        
        Sinks and Local history are fully visible (unmasked).
        Causal self (q_len x q_len) is causally masked.
        """
        total_k = k_sink + k_local + q_len
        mask = torch.ones((q_len, total_k), dtype=torch.bool, device=device)
        
        # Causal mask for the chunk's self-attention
        causal = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=device))
        
        # Assemble: [Full Sinks | Full Local | Causal Self]
        mask[:, -(q_len):] = causal
        return mask

    def execute_sparse_attention(
        self,
        q: torch.Tensor,  # [bsz, heads, seq_len, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes sparse chunked prefill attention.
        Assumes q, k, v are the full sequence projections, but we will chunk
        their processing to save memory traffic and FLOPs.
        """
        bsz, heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        
        # If sequence is small, just do dense SDPA
        if seq_len <= self.chunk_size + self.sink_tokens:
            self.stats["dense_flops"] += seq_len * seq_len * head_dim
            self.stats["sparse_flops"] += seq_len * seq_len * head_dim
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = torch.zeros_like(q)
        
        # Pre-extract sink KV
        k_sink = k[:, :, :self.sink_tokens, :]
        v_sink = v[:, :, :self.sink_tokens, :]
        
        # Compute dense flops for reporting (QxK + AttnxV)
        self.stats["dense_flops"] += 2 * bsz * heads * (seq_len * seq_len * head_dim)
        
        t0 = time.perf_counter()
        
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        for c in range(num_chunks):
            start_idx = c * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            q_chunk = q[:, :, start_idx:end_idx, :]
            
            # Local window KV bounds
            local_start = max(self.sink_tokens, start_idx - (self.local_window_chunks * self.chunk_size))
            local_end = start_idx
            
            # Gather K and V for this chunk
            if local_start < local_end:
                k_local = k[:, :, local_start:local_end, :]
                v_local = v[:, :, local_start:local_end, :]
                k_context = torch.cat([k_sink, k_local, k[:, :, start_idx:end_idx, :]], dim=2)
                v_context = torch.cat([v_sink, v_local, v[:, :, start_idx:end_idx, :]], dim=2)
                
                # Build custom mask
                q_len = q_chunk.shape[2]
                mask = self._create_chunk_mask(q_len, k_sink.shape[2], k_local.shape[2], dtype, device)
                
                # Mask must be [bsz, heads, q_len, k_len] for SDPA
                # Wait, SDPA accepts boolean masks in shape [..., q_len, k_len]
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                out_chunk = F.scaled_dot_product_attention(q_chunk, k_context, v_context, attn_mask=mask)
                
            else:
                # Early chunks where local window overlaps with sinks or is empty
                k_context = torch.cat([k_sink, k[:, :, start_idx:end_idx, :]], dim=2)
                v_context = torch.cat([v_sink, v[:, :, start_idx:end_idx, :]], dim=2)
                
                q_len = q_chunk.shape[2]
                mask = self._create_chunk_mask(q_len, k_sink.shape[2], 0, dtype, device)
                mask = mask.unsqueeze(0).unsqueeze(0)
                
                out_chunk = F.scaled_dot_product_attention(q_chunk, k_context, v_context, attn_mask=mask)
                
            out[:, :, start_idx:end_idx, :] = out_chunk
            
            # Accumulate sparse FLOPs
            k_len = k_context.shape[2]
            self.stats["sparse_flops"] += 2 * bsz * heads * (q_len * k_len * head_dim)
            
        t1 = time.perf_counter()
        self.stats["time_ms"] = (t1 - t0) * 1000
        
        return out

    def get_summary(self) -> dict:
        flops_saved = self.stats["dense_flops"] - self.stats["sparse_flops"]
        ratio = (self.stats["sparse_flops"] / max(1, self.stats["dense_flops"]))
        return {
            "dense_gflops": round(self.stats["dense_flops"] / 1e9, 2),
            "sparse_gflops": round(self.stats["sparse_flops"] / 1e9, 2),
            "flops_reduced_pct": round((1.0 - ratio) * 100, 2),
            "execution_time_ms": round(self.stats["time_ms"], 2)
        }
