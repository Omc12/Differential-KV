"""
test_phase15_orchestration.py

Profiles the Sparse Prefill Anchor Engine to identify orchestration hotspots
and test grouped dispatch/fusion solutions.
"""

import sys, time, torch
from torch.profiler import profile, record_function, ProfilerActivity
sys.path.insert(0, ".")

from runtime.sparse_prefill_anchors import RetrievalAwareSparsePrefill

DEVICE = "cuda"
BSZ = 1
HEADS = 8
SEQ_LEN = 16384
HEAD_DIM = 128
CHUNK_SIZE = 512

torch.manual_seed(42)
q = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16)
k = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16)
v = torch.randn(BSZ, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=torch.float16)

engine = RetrievalAwareSparsePrefill(sink_tokens=64, chunk_size=CHUNK_SIZE, local_window_chunks=1, top_k_retrieval_chunks=1)

# Warmup
_ = engine.execute_sparse_attention(q, k, v)
torch.cuda.synchronize()

print("Profiling...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("sparse_prefill_anchors"):
        _ = engine.execute_sparse_attention(q, k, v)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
