"""
test_phase10_cuda_graph.py

Phase 10: CUDA Graph Reality Validation
Tests whether the new Native Block Pool architecture allows the entire
HuggingFace layer decode to be captured and replayed via CUDA Graphs, 
bypassing the 8.7ms PyTorch CPU overhead entirely.
"""

import sys, time, torch
from typing import Optional
sys.path.insert(0, ".")

from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from runtime.native_block_pool import NativeBlockPool
from native_core.sparse_decode.triton_sparse_attn import native_triton_sparse_attn_decode
from native_core.kv_runtime_manager import KVBlock
from native_core.compression.lowrank import compress_lowrank

def make_compressed_block(seed=0):
    torch.manual_seed(seed)
    k = torch.randn(1, 4, 64, 128, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, 4, 64, 128, device=DEVICE, dtype=torch.float16)
    anchor_kv = torch.stack([k[:, :, 0], v[:, :, 0]], dim=1)
    blk = KVBlock(anchor_idx=0, anchor_kv=anchor_kv, token_indices=list(range(64)))

    feat_dim  = 2 * 4 * 128
    stacked   = torch.stack([k[0, :, 1:].transpose(0, 1), v[0, :, 1:].transpose(0, 1)], dim=1)
    flat      = stacked.reshape(63, feat_dim).float()
    anc_flat  = anchor_kv.view(-1).float()
    deltas    = flat - anc_flat.unsqueeze(0)
    lr        = compress_lowrank(deltas, rank=16)
    blk.U, blk.V, blk.scale = lr.U, lr.V, lr.scale
    return blk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("PHASE 10 — CUDA GRAPH E2E LAYER DECODE VALIDATION")
print("=" * 60)

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
config.num_hidden_layers = 1
model = Qwen2ForCausalLM(config).to(DEVICE).to(torch.float16).eval()

# Native pool
pool = NativeBlockPool(max_blocks=256, num_kv_heads=4, head_dim=128, rank=16, max_seq_len=64, device=DEVICE)

# Patch the HF layer manually for the test to use NativeBlockPool directly
# bypassing KVRuntimeManager (which has Python dynamic state logic)
layer = model.model.layers[0]
original_forward = layer.forward

def static_diffkv_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[torch.Tensor] = None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    
    # 1. Projections
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, 28, 128).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, 4, 128).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, 4, 128).transpose(1, 2)
    
    # 3. Sparse Triton Kernel with NativeBlockPool
    block_indices = self.static_block_indices  # Hack: pass block_indices through layer property
    
    attn_output = native_triton_sparse_attn_decode(
        q=query_states,
        block_indices=block_indices,
        pool=pool,
        dense_blocks=[],
        active_k=None,
        active_v=None,
        num_key_value_groups=7,
        R=16,
        S_MAX=64
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, 28 * 128)
    attn_output = self.o_proj(attn_output)
    
    return attn_output, None

# Apply patch
import types
model.model.layers[0].self_attn.forward = types.MethodType(static_diffkv_attention_forward, model.model.layers[0].self_attn)

# Prepare inputs
N = 64
block_indices = []
for i in range(N):
    blk = make_compressed_block(i)
    pool_idx = pool.allocate_block()
    pool.write_block(pool_idx, blk.U, blk.V, blk.anchor_kv[:, 0], blk.anchor_kv[:, 1], blk.scale, blk.U.shape[0])
    block_indices.append(pool_idx)

static_block_indices = torch.tensor(block_indices, device=DEVICE, dtype=torch.int32)
model.model.layers[0].self_attn.static_block_indices = static_block_indices

# Graph Inputs
static_hidden = torch.randn(1, 1, config.hidden_size, device=DEVICE, dtype=torch.float16)
static_pos_id = torch.tensor([[4096]], device=DEVICE)

# 1. Warmup Eager
print("Warming up eager execution...")
for _ in range(5):
    with torch.no_grad():
        out_eager = layer(
            hidden_states=static_hidden,
            position_ids=static_pos_id,
            use_cache=False
        )
torch.cuda.synchronize()

# 2. Capture CUDA Graph
print("Capturing CUDA Graph...")
g = torch.cuda.CUDAGraph()

# Need to do one warmup before capture to initialize pools
with torch.no_grad():
    layer(hidden_states=static_hidden, position_ids=static_pos_id, use_cache=False)
torch.cuda.synchronize()

with torch.cuda.graph(g):
    out_graph = layer(
        hidden_states=static_hidden,
        position_ids=static_pos_id,
        use_cache=False
    )

print("Capture successful!")

# 3. Timing Comparison (Whole loop timing to hide WDDM command buffer overhead)
STEPS = 100

# Eager
torch.cuda.synchronize()
t0_eager = time.perf_counter()
with torch.no_grad():
    for i in range(STEPS):
        layer(hidden_states=static_hidden, position_ids=static_pos_id, use_cache=False)
torch.cuda.synchronize()
t1_eager = time.perf_counter()
avg_eager = ((t1_eager - t0_eager) * 1000) / STEPS

# Graph
torch.cuda.synchronize()
t0_graph = time.perf_counter()
with torch.no_grad():
    for i in range(STEPS):
        g.replay()
torch.cuda.synchronize()
t1_graph = time.perf_counter()
avg_graph = ((t1_graph - t0_graph) * 1000) / STEPS

print("\n== Eager Python vs CUDA Graph E2E Layer Latency ==")
print(f"  Eager (Python orchestrates) : {avg_eager:.3f} ms")
print(f"  Graph (GPU replays static)  : {avg_graph:.3f} ms")
print(f"  Speedup                     : {avg_eager / avg_graph:.1f}x")

print("\n" + "=" * 60)
print("PHASE 10 CUDA GRAPH VALIDATION COMPLETE")
