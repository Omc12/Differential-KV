
import torch
import time
from typing import Dict, Any, List, Optional, Tuple

from krx.fused_sparse_attention_kernel import FusedSparseAttentionKernel
from krx.activation_memory_compressor import ActivationMemoryCompressor
from krx.execution_path_prefetcher import ExecutionPathPrefetcher
from krx.kernel_legitimacy_guard import KernelLegitimacyGuard

class SparseKernelDispatcher:
    """
    PHASE 23.0: KRX - Sparse Kernel Dispatcher.
    The primary entry point for sparse-native execution acceleration.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda")
        
        # KRX Components
        self.fused_attention = FusedSparseAttentionKernel(config)
        self.memory_compressor = ActivationMemoryCompressor(config)
        self.prefetcher = ExecutionPathPrefetcher(config)
        self.guard = KernelLegitimacyGuard(config)
        
        self.metrics = {
            "kernel_acceleration_gain": 0.0,
            "memory_compression_ratio": 0.0,
            "prefetch_accuracy": 0.0,
            "sparse_kernel_stability": 1.0,
            "symbolic_continuity": 1.0,
            "execution_entropy_health": 1.0
        }

    def dispatch_attention(self, 
                           q: torch.Tensor, 
                           k: torch.Tensor, 
                           v: torch.Tensor, 
                           mask: Optional[torch.Tensor] = None,
                           symbolic_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dispatches a fused sparse attention call with prefetching and compression.
        """
        # 1. Prefetching
        self.prefetcher.predict_and_prefetch("attention_block", ["attention_block", "ffn_block"])
        
        # 2. Execution (Fused Sparse Attention)
        output = self.fused_attention.execute(q, k, v, mask, symbolic_weights)
        
        # 3. Guarding
        is_legit = self.guard.validate_execution(q, output, "fused_sparse_attention")
        if not is_legit:
            # Fallback or correction logic could go here
            pass
            
        # 4. Activation Compression (Post-execution)
        compressed_output = self.memory_compressor.compress_activations(output, "attn_out")
        
        self._update_aggregate_metrics()
        
        return compressed_output

    def _update_aggregate_metrics(self):
        """Aggregates metrics from all KRX sub-components."""
        attn_m = self.fused_attention.get_metrics()
        mem_m = self.memory_compressor.get_metrics()
        pref_m = self.prefetcher.get_metrics()
        guard_m = self.guard.get_metrics()
        
        self.metrics["kernel_acceleration_gain"] = attn_m["kernel_acceleration_gain"]
        self.metrics["memory_compression_ratio"] = mem_m["memory_compression_ratio"]
        self.metrics["prefetch_accuracy"] = pref_m["prefetch_accuracy"]
        self.metrics["sparse_kernel_stability"] = guard_m["sparse_kernel_stability"]
        self.metrics["symbolic_continuity"] = attn_m.get("symbolic_continuity_preserved", 1.0)
        self.metrics["execution_entropy_health"] = attn_m["execution_entropy_health"]

    def get_metrics(self) -> Dict[str, Any]:
        self._update_aggregate_metrics()
        return self.metrics
