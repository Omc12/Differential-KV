# Phase 28 — Native Block Pool Activation

## Objective
Activate `NativeBlockPool` in the live serving path. Replace pure Python `torch.stack` block structures with a pre-allocated contiguous memory pool (vLLM style block tables) to allow native kernel dispatch.

## Action Taken
1. **Instantiation**: 
   - Initialized `NativeBlockPool` within `KVRuntimeManager.__init__` with a capacity of `16384` blocks, `num_kv_heads`, `head_dim`, `rank=32`, and `max_seq_len=64` to match the model and compression specs.
2. **Data Routing**:
   - Modified the `KVBlock` dataclass to store a `pool_idx`.
   - Updated `_compress_block_sync` in `KVRuntimeManager` to automatically allocate a block index via `self.native_pool.allocate_block()` upon successful SVD compression.
   - Called `self.native_pool.write_block()` to transfer the newly computed `U`, `V`, `anchor_K`, `anchor_V`, and metadata directly into the pre-allocated GPU pool.
3. **Integration**:
   - This change fundamentally replaces the previous architecture where Python managed individual tensors. It ensures that compressed tokens seamlessly transition to contiguous memory, fulfilling the prerequisite for Triton fused sparse decode.

**Status**: SUCCESS
