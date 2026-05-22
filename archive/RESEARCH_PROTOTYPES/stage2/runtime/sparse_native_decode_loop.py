"""
Sparse-Native Decode Loop

Reduces Python orchestration overhead and maintains persistent autoregressive execution.
"""
import torch

class SparseNativeDecodeLoop:
    def __init__(self, attention_engine, kernel_runtime, residency_layer):
        self.attention_engine = attention_engine
        self.kernel_runtime = kernel_runtime
        self.residency_layer = residency_layer
        
    def persistent_decode_step(self, input_ids, kv_cache_ptr):
        """
        Executes a decode step with minimized tensor recreation and CPU synchronization.
        """
        # 1. Residency lookup (no dense reconstruction)
        sparse_k, sparse_v, indices = self.residency_layer.get_resident_blocks(kv_cache_ptr)
        
        # 2. Persistent kernel dispatch
        q = self.kernel_runtime.prepare_q_persistent(input_ids)
        
        # 3. Sparse-native attention computation
        out, metrics = self.attention_engine.execute_sparse_attention(
            q, sparse_k, sparse_v, indices, metadata={}
        )
        
        return out, metrics
