import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from native_core.compression.lowrank import compress_lowrank

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )
    
    # We patch _compress_block_sync to use Channel Norm-Normalized SVD
    manager = wrapper.manager
    
    def patched_compress_block_sync(self, block, k, v):
        input_device = k.device
        if input_device.type == "cpu":
            anchor_kv_local = getattr(block, "anchor_kv_cpu", None)
            if anchor_kv_local is None:
                anchor_kv_local = block.anchor_kv.cpu()
        else:
            anchor_kv_local = block.anchor_kv
        anchor_flat = anchor_kv_local.reshape(-1).float().to(input_device)
        seq_len  = k.shape[2]
        heads    = k.shape[1]
        head_dim = k.shape[3]
        feat_dim = 2 * heads * head_dim

        stacked     = torch.stack([k[0].transpose(0, 1), v[0].transpose(0, 1)], dim=1)
        flat_tokens = stacked.reshape(seq_len, feat_dim).float()
        deltas      = flat_tokens - anchor_flat.unsqueeze(0)

        # Norm-Normalization
        channel_norms = deltas.norm(dim=0)
        channel_norms = torch.clamp(channel_norms, min=1e-5)
        normalized_deltas = deltas / channel_norms.unsqueeze(0)

        rank = self.rank
        lr_delta = compress_lowrank(normalized_deltas, rank)
        
        V_scaled = lr_delta.V.float() * channel_norms.unsqueeze(0)
        V_scaled = V_scaled.to(torch.float16)

        gpu_device = block.anchor_kv.device
        block.U          = lr_delta.U.to(gpu_device)
        block.V          = V_scaled.to(gpu_device)
        block.scale      = lr_delta.scale
        block.cosine_sim = lr_delta.cosine_sim
        block.norm_drift = lr_delta.norm_drift
        block.dynamic_rank = getattr(lr_delta, "dynamic_rank", self.rank)

        block.active_k = None
        block.active_v = None
        block.active_k_cpu = None
        block.active_v_cpu = None
        block.dirty    = True
        block.state = "COMPRESSED"

        session_id = getattr(block, 'session_id', None)
        session_active = True
        if session_id is not None:
            if self._streaming_mgr is not None:
                session_active = session_id in self._streaming_mgr.session_blocks
            else:
                session_active = session_id in self.session_blocks

        if session_active:
            if hasattr(self, 'native_pool') and self.native_pool is not None:
                try:
                    if getattr(block, 'pool_idx', None) is None:
                        block.pool_idx = self.native_pool.allocate_block()
                    self.native_pool.write_block(
                        pool_idx=block.pool_idx,
                        U=block.U,
                        V=block.V,
                        anchor_K=block.anchor_kv[0, 0],
                        anchor_V=block.anchor_kv[0, 1],
                        scale=block.scale,
                        seq_len=block.U.shape[0]
                    )
                except Exception as e:
                    print(f"[DiffKV] WARNING: Failed to write block to NativeBlockPool: {e}")

            if self._streaming_mgr is not None and getattr(block, 'session_id', None) is not None and getattr(block, 'layer_idx', None) is not None:
                self._streaming_mgr.update_metadata_state(block.session_id, block.layer_idx, block)

    import types
    manager._compress_block_sync = types.MethodType(patched_compress_block_sync, manager)

    # Let's patch the region partitioning in ingest_chunk of StreamingSparseIngestManager
    original_ingest_chunk = manager._streaming_mgr.ingest_chunk
    
    def patched_ingest_chunk(self, session_id, layer_idx, k, v):
        # We temporarily patch self.micro_block_size or the regions
        # To inspect and modify regions dynamically, let's write a wrapper
        seq_len = k.shape[2]
        if seq_len > 1:
            # Prefill path
            # We partition the prefill sequence using micro_block_size for all regions!
            micro_block_size = self._streaming_mgr.session_micro_block_sizes.get(session_id, self._streaming_mgr.micro_block_size)
            
            regions = []
            r1_start = max(0, seq_len - 1024)
            if r1_start < seq_len:
                regions.append((r1_start, seq_len, micro_block_size))
            r2_start = max(0, seq_len - 4096)
            if r2_start < r1_start:
                regions.append((r2_start, r1_start, micro_block_size))
            r3_start = max(0, seq_len - 12288)
            if r3_start < r2_start:
                regions.append((r3_start, r2_start, micro_block_size))
            if 0 < r3_start:
                regions.append((0, r3_start, micro_block_size))
            regions.reverse()
            
            # Now we replicate the rest of the ingest_chunk prefill path, or we can just modify the regions and call it!
            # Since the original ingest_chunk has hardcoded regions, we can't easily run it directly.
            # But wait! We can just modify the original file, it is much easier and cleaner!
            
        return original_ingest_chunk(session_id, layer_idx, k, v)
        
    manager._streaming_mgr.ingest_chunk = types.MethodType(patched_ingest_chunk, manager._streaming_mgr)

    # Actually, let's modify the original file ACTIVE_RUNTIME/native_core/streaming_sparse_ingest.py directly!
    # That is much cleaner and more permanent.
    
    # We will modify the regions in streaming_sparse_ingest.py.
    
if __name__ == "__main__":
    print("Pre-modification analysis done.")
