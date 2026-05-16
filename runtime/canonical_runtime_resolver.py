"""
Canonical Runtime Resolver

Authoritative entrypoint for Differential KV execution.
Selects sparse-native paths while preserving Stage 1 safety and compatibility.
"""

from runtime.execution.sparse_native_decode_loop import SparseNativeDecodeLoop
from runtime.sparse.sparse_native_attention_engine import SparseNativeAttentionEngine
from runtime.kernels.persistent_sparse_kernel_runtime import PersistentSparseKernelRuntime
from runtime.sparse.sparse_tensor_residency_layer import SparseTensorResidencyLayer

class CanonicalRuntimeResolver:
    def __init__(self, mode="sparse-native"):
        self.mode = mode
        self.engine = SparseNativeAttentionEngine()
        self.runtime = PersistentSparseKernelRuntime()
        self.residency = SparseTensorResidencyLayer()
        self.decode_loop = SparseNativeDecodeLoop(self.engine, self.runtime, self.residency)
        
    def resolve_execution_path(self, request):
        """
        Preserves OpenAI and WebUI compatibility while using sparse-native paths.
        """
        if self.mode == "sparse-native":
            return self.decode_loop
        else:
            # Stage 1 Fallback placeholder
            return None

    def handle_request(self, input_ids):
        """
        Canonical entry point for inference.
        """
        return self.decode_loop.persistent_decode_step(input_ids, kv_cache_ptr=None)
