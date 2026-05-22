"""
runtime/fused_sparse_prefill.py

Phase 15 — Runtime Fusion & Orchestration Collapse

Replaces the CPU-bound, loop-heavy chunked sparse prefill with a single fused 
kernel utilizing PyTorch flex_attention and pre-routed global anchors.

Orchestration Collapse:
1. Eliminates 32+ Python chunk loops.
2. Eliminates 32+ torch.cat() and masking operations.
3. Eliminates 32+ SDPA kernel launches.
4. Pre-routes all anchors natively in a single batched GEMM.
5. Executes the entire 16K+ sequence in a SINGLE fused attention kernel.

This is production-ready sparse transformer execution.
"""

import torch
import time
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False

class FusedSparsePrefill:
    def __init__(
        self,
        sink_tokens: int = 64,
        chunk_size: int = 512,
        local_chunks: int = 1,
        top_k_retrieval: int = 1
    ):
        self.sink_tokens = sink_tokens
        self.chunk_size = chunk_size
        self.local_chunks = local_chunks
        self.top_k_retrieval = top_k_retrieval
        
        self.stats = {
            "routing_time_ms": 0.0,
            "mask_time_ms": 0.0,
            "attn_time_ms": 0.0,
            "total_time_ms": 0.0
        }

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not FLEX_AVAILABLE:
            raise RuntimeError("flex_attention requires PyTorch 2.5+")
            
        bsz, heads, seq_len, head_dim = q.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        t0 = time.perf_counter()
        
        # ── 1. PRE-ROUTE GLOBAL ANCHORS ───────────────────────────────────────
        t_route_0 = time.perf_counter()
        
        # Truncate to exact multiples of chunk_size for pooling (or pad, but we assume exact for now)
        assert seq_len % self.chunk_size == 0, "Seq len must be multiple of chunk size for fused engine"
        
        # [bsz, heads, num_chunks, chunk_size, head_dim]
        q_chunks = q.view(bsz, heads, num_chunks, self.chunk_size, head_dim)
        k_chunks = k.view(bsz, heads, num_chunks, self.chunk_size, head_dim)
        
        # Centroid pooling -> [bsz, heads, num_chunks, head_dim]
        q_anchors = q_chunks.mean(dim=3)
        k_anchors = k_chunks.mean(dim=3)
        
        # Relevance: [bsz, heads, num_chunks, num_chunks]
        relevance = torch.matmul(q_anchors, k_anchors.transpose(-2, -1))
        
        # Mask out future chunks and local/sink chunks so we only retrieve distant past
        # Local window is `local_chunks`. So for query chunk C, chunks >= C - local_chunks are local/future
        causal_mask = torch.ones((num_chunks, num_chunks), dtype=torch.bool, device=q.device)
        causal_mask = torch.tril(causal_mask, diagonal=-(self.local_chunks + 1))
        
        relevance.masked_fill_(~causal_mask, float('-inf'))
        
        # Pool relevance across heads to find globally relevant chunks (optional, we do it for simplicity)
        global_relevance = relevance.mean(dim=(0, 1)) # [num_chunks, num_chunks]
        
        chunk_mask = torch.zeros((num_chunks, num_chunks), dtype=torch.bool, device=q.device)
        
        if self.top_k_retrieval > 0:
            # We can't topk if there are no searchable chunks, handled by -inf mask
            _, top_idx = torch.topk(global_relevance, min(self.top_k_retrieval, num_chunks), dim=1)
            # Scatter to boolean mask
            chunk_mask.scatter_(1, top_idx, True)
            
        # Add back Sinks and Local windows
        sink_chunks = max(1, self.sink_tokens // self.chunk_size)
        chunk_mask[:, :sink_chunks] = True
        
        # Add local window and self
        local_mask = torch.tril(torch.ones((num_chunks, num_chunks), dtype=torch.bool, device=q.device))
        local_mask = local_mask & ~torch.tril(local_mask, diagonal=-(self.local_chunks + 1))
        
        final_chunk_mask = chunk_mask | local_mask
        
        # Convert to flat tensor for compiler capture
        # shape [num_chunks, num_chunks]
        
        t_route_1 = time.perf_counter()
        
        # ── 2. COMPILE BLOCK MASK ─────────────────────────────────────────────
        # flex_attention block mask function
        def sparse_mask_mod(b, h, q_idx, kv_idx):
            q_c = q_idx // self.chunk_size
            kv_c = kv_idx // self.chunk_size
            
            # 1. Must be causal
            is_causal = q_idx >= kv_idx
            # 2. Must be selected in our chunk mask
            is_selected = final_chunk_mask[q_c, kv_c]
            
            return is_causal & is_selected

        # create_block_mask compiles the sparsity pattern into a block-level metadata tensor
        block_mask = create_block_mask(sparse_mask_mod, bsz, heads, seq_len, seq_len)
        
        t_mask_1 = time.perf_counter()
        
        # ── 3. FUSED ATTENTION ────────────────────────────────────────────────
        # Must compile flex_attention to prevent OOM
        if not hasattr(self, "_compiled_flex_attention"):
            self._compiled_flex_attention = torch.compile(flex_attention)
        
        out = self._compiled_flex_attention(q, k, v, block_mask=block_mask)
        
        torch.cuda.synchronize()
        t_attn_1 = time.perf_counter()
        
        self.stats["routing_time_ms"] = (t_route_1 - t_route_0) * 1000
        self.stats["mask_time_ms"] = (t_mask_1 - t_route_1) * 1000
        self.stats["attn_time_ms"] = (t_attn_1 - t_mask_1) * 1000
        self.stats["total_time_ms"] = (t_attn_1 - t0) * 1000
        
        return out
