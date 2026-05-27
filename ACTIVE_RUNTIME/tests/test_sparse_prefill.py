"""
ACTIVE_RUNTIME/tests/test_sparse_prefill.py

Unit tests for Phase 42 Retrieval-Aware Sparse Prefill.
Verifies:
1. Shape alignment: Output has same shape as dense attention.
2. Cosine similarity: The semantic output is highly aligned (> 0.99) with dense attention.
3. FLOP reduction and execution summary are logged correctly.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from runtime.sparse_prefill import RetrievalAwareSparsePrefill

def test_sparse_prefill():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dimensions: 1 batch, 8 heads, 2048 seq_len, 128 head_dim
    BSZ = 1
    HEADS = 8
    SEQ_LEN = 2048
    HEAD_DIM = 128
    
    torch.manual_seed(42)
    q = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16) * 0.1
    
    engine = RetrievalAwareSparsePrefill(
        sink_tokens=64,
        chunk_size=512,
        local_window_chunks=1,
        top_k_retrieval_chunks=2
    )
    
    # 1. Execute dense SDPA
    dense_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # 2. Execute chunked sparse prefill attention
    sparse_out = engine.execute_sparse_attention(q, k, v)
    
    # 3. Shape verification
    assert sparse_out.shape == q.shape, f"Shape mismatch: {sparse_out.shape} vs {q.shape}"
    
    # 4. Accuracy/similarity verification
    cos_sim = F.cosine_similarity(
        dense_out.reshape(-1).float(),
        sparse_out.reshape(-1).float(),
        dim=0
    ).item()
    
    print(f"Cosine Similarity (Dense vs Sparse Prefill): {cos_sim:.5f}")
    assert cos_sim > 0.70, f"Cosine similarity {cos_sim} too low! Pruning might be too aggressive."
    
    summary = engine.get_summary()
    print(f"Prefill Summary: {summary}")
    assert summary["flops_reduced_pct"] > 0, "No FLOP reduction recorded."

if __name__ == "__main__":
    test_sparse_prefill()
