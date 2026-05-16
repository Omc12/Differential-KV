import torch
import os
from typing import Dict, Any, Optional

from aggressive_kv_materializer import AggressiveKVMaterializer
from hard_attention_pruner import HardAttentionPruner
from sparse_flop_accountant import accountant
from runtime_density_profiler import profiler
from sparse_participation_controller import controller
from sem_integrity_guard import SEMIntegrityGuard
from runtime.kv_runtime_manager import KVRuntimeManager

class SEMResolver:
    """
    Main resolver for SEM (Sparse Economics Materialization).
    Coordinates eviction, pruning, accounting, and profiling.
    """
    def __init__(self, manager: KVRuntimeManager):
        self.manager = manager
        self.aggressive_mode = os.environ.get("DIFFKV_AGGRESSIVE_SPARSE_MODE") == "1"
        
        self.materializer = AggressiveKVMaterializer(manager, aggressive_mode=self.aggressive_mode)
        self.pruner = HardAttentionPruner(
            top_k_ratio=controller.config["max_participation_ratio"],
            aggressive_mode=self.aggressive_mode
        )
        self.guard = SEMIntegrityGuard()
        
        print(f"[SEM] Resolver initialized (Aggressive Mode: {self.aggressive_mode})")

    def resolve_attention(
        self, 
        layer_idx: int, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        curvature: Optional[torch.Tensor] = None
    ):
        """
        Executes sparse attention with materialization accounting.
        """
        profiler.start("attention")
        
        # 1. Apply KV Eviction Pressure
        self.materializer.apply_eviction_pressure(layer_idx)
        
        # 2. Hard Attention Pruning
        k_sparse, v_sparse = self.pruner.prune_attention(q, k, v, curvature)
        
        # 3. Real FLOP Accounting
        bsz, n_heads, q_len, d = q.shape
        seq_len = k.shape[2]
        sparse_len = k_sparse.shape[2]
        accountant.record_attention(q_len, seq_len, d, n_heads, sparse_len)
        
        profiler.end("attention")
        return k_sparse, v_sparse

    def finalize_step(self):
        """
        Finalizes a decode step and validates integrity.
        """
        # Sampling and logits profiling are handled externally in the main loop
        pass

    def get_sem_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive SEM economics report.
        """
        kv_metrics = self.materializer.get_residency_report()
        flop_metrics = accountant.get_metrics()
        density_report = profiler.get_report()
        
        report = {
            **kv_metrics,
            **flop_metrics,
            **density_report,
            "aggressive_mode": self.aggressive_mode
        }
        
        self.guard.validate_metrics(report)
        self.guard.check_integrity()
        
        return report
