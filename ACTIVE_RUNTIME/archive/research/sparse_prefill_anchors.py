"""
runtime/sparse_prefill_anchors.py

Phase 14 — Retrieval-Aware Sparse Execution & Global Memory Anchors

Implements Global Memory Anchors to restore retrieval capacity in sparse prefill.
Locality-only sparse prefill loses historical information. This module uses chunk
centroids as "anchors" to maintain a global semantic map.

Mechanism:
1. When a chunk is processed, its anchor (mean of Keys) is saved to an Anchor Pool.
2. For each new chunk, we first compute relevance between the chunk's Queries and
   the global Anchor Pool.
3. The top-K most relevant historical chunks are temporarily paged back into the 
   attention window (Retrieval-Aware Routing).
4. The SDPA is executed over: Sinks + Top-K Historical Chunks + Local Window + Self.

This restores O(1) global retrieval visibility while maintaining O(N) prefill scaling.
"""

import torch
import torch.nn.functional as F
import time

class RetrievalAwareSparsePrefill:
    """
    Executes sparse prefill with global anchor routing to preserve retrieval.
    """
    def __init__(
        self,
        sink_tokens: int = 64,
        chunk_size: int = 512,
        local_window_chunks: int = 1,
        top_k_retrieval_chunks: int = 1
    ):
        self.sink_tokens = sink_tokens
        self.chunk_size = chunk_size
        self.local_window_chunks = local_window_chunks
        self.top_k_retrieval_chunks = top_k_retrieval_chunks
        
        self.stats = {
            "dense_flops": 0,
            "sparse_flops": 0,
            "routing_flops": 0,
            "time_ms": 0.0,
            "retrievals": 0
        }

    def _create_chunk_mask(self, q_len: int, k_sink: int, k_retrieved: int, k_local: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Creates the attention mask for a chunk.
        [ Full Sinks | Full Retrieved | Full Local | Causal Self ]
        """
        total_k = k_sink + k_retrieved + k_local + q_len
        mask = torch.ones((q_len, total_k), dtype=torch.bool, device=device)
        
        causal = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=device))
        mask[:, -(q_len):] = causal
        return mask

    def execute_sparse_attention(
        self,
        q: torch.Tensor,  # [bsz, heads, seq_len, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        bsz, heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        
        # Dense fallback for small sequences
        if seq_len <= self.chunk_size * 3:
            self.stats["dense_flops"] += seq_len * seq_len * head_dim
            self.stats["sparse_flops"] += seq_len * seq_len * head_dim
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = torch.zeros_like(q)
        
        k_sink = k[:, :, :self.sink_tokens, :]
        v_sink = v[:, :, :self.sink_tokens, :]
        
        self.stats["dense_flops"] += 2 * bsz * heads * (seq_len * seq_len * head_dim)
        
        t0 = time.perf_counter()
        
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Global Anchor Pool
        anchor_keys = []
        chunk_kv_refs = [] # Keep refs to the chunks for retrieval
        
        for c in range(num_chunks):
            start_idx = c * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            q_chunk = q[:, :, start_idx:end_idx, :]
            k_chunk = k[:, :, start_idx:end_idx, :]
            v_chunk = v[:, :, start_idx:end_idx, :]
            
            # Local window bounds
            local_start = max(self.sink_tokens, start_idx - (self.local_window_chunks * self.chunk_size))
            local_end = start_idx
            
            k_retrieved_list = []
            v_retrieved_list = []
            k_retrieved_len = 0
            
            # ── ANCHOR ROUTING ────────────────────────────────────────────────
            # Only search if we have anchors that are outside the local window
            # (Chunks before local_start)
            searchable_chunks = (local_start - self.sink_tokens) // self.chunk_size
            
            if searchable_chunks > 0 and len(anchor_keys) >= searchable_chunks:
                # We can search the anchors up to searchable_chunks
                # [bsz, heads, searchable, head_dim]
                K_anchors = torch.stack(anchor_keys[:searchable_chunks], dim=2)
                
                # To reduce routing cost, we pool Q_chunk into a single query vector per head
                # [bsz, heads, 1, head_dim]
                Q_anchor = q_chunk.mean(dim=2, keepdim=True)
                
                # Compute relevance: Q_anchor @ K_anchors^T -> [bsz, heads, 1, searchable]
                relevance = torch.matmul(Q_anchor, K_anchors.transpose(-2, -1))
                
                # Average relevance across heads to find globally relevant chunks (or route per head)
                # To save memory traffic, we route globally across heads (mean over heads)
                global_relevance = relevance.mean(dim=(0, 1, 2)) # [searchable]
                
                # Select top-k chunks
                k_retrieve = min(self.top_k_retrieval_chunks, searchable_chunks)
                if k_retrieve > 0:
                    _, top_indices = torch.topk(global_relevance, k_retrieve)
                    
                    for idx in top_indices.tolist():
                        hist_k, hist_v = chunk_kv_refs[idx]
                        k_retrieved_list.append(hist_k)
                        v_retrieved_list.append(hist_v)
                        k_retrieved_len += hist_k.shape[2]
                        self.stats["retrievals"] += 1
                        
                self.stats["routing_flops"] += 2 * bsz * heads * (1 * searchable_chunks * head_dim)
            
            # ── ASSEMBLE CONTEXT ──────────────────────────────────────────────
            k_local_len = 0
            
            parts_k = [k_sink]
            parts_v = [v_sink]
            
            if k_retrieved_list:
                parts_k.extend(k_retrieved_list)
                parts_v.extend(v_retrieved_list)
                
            if local_start < local_end:
                k_local = k[:, :, local_start:local_end, :]
                v_local = v[:, :, local_start:local_end, :]
                parts_k.append(k_local)
                parts_v.append(v_local)
                k_local_len = k_local.shape[2]
                
            parts_k.append(k_chunk)
            parts_v.append(v_chunk)
            
            k_context = torch.cat(parts_k, dim=2)
            v_context = torch.cat(parts_v, dim=2)
            
            q_len = q_chunk.shape[2]
            mask = self._create_chunk_mask(q_len, self.sink_tokens, k_retrieved_len, k_local_len, dtype, device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            
            out_chunk = F.scaled_dot_product_attention(q_chunk, k_context, v_context, attn_mask=mask)
            out[:, :, start_idx:end_idx, :] = out_chunk
            
            self.stats["sparse_flops"] += 2 * bsz * heads * (q_len * k_context.shape[2] * head_dim)
            
            # ── GENERATE ANCHOR FOR THIS CHUNK ────────────────────────────────
            # Mean-pooling over the sequence dimension to create a centroid anchor
            anchor_k = k_chunk.mean(dim=2) # [bsz, heads, head_dim]
            anchor_keys.append(anchor_k)
            chunk_kv_refs.append((k_chunk, v_chunk)) # Save for potential retrieval
            
        t1 = time.perf_counter()
        self.stats["time_ms"] = (t1 - t0) * 1000
        
        return out

    def get_summary(self) -> dict:
        total_sparse = self.stats["sparse_flops"] + self.stats["routing_flops"]
        ratio = (total_sparse / max(1, self.stats["dense_flops"]))
        return {
            "dense_gflops": round(self.stats["dense_flops"] / 1e9, 2),
            "sparse_gflops": round(total_sparse / 1e9, 2),
            "routing_gflops": round(self.stats["routing_flops"] / 1e9, 4),
            "flops_reduced_pct": round((1.0 - ratio) * 100, 2),
            "execution_time_ms": round(self.stats["time_ms"], 2),
            "total_retrieval_events": self.stats["retrievals"]
        }
