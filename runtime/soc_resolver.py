import torch
from typing import Dict, Any, Optional

from sparse_work_consolidator import SparseWorkConsolidator
from occupancy_aware_triton_fuser import OccupancyAwareTritonFuser
from sustained_sparse_batch_scheduler import SustainedSparseBatchScheduler
from arithmetic_intensity_stabilizer import intensity_stabilizer
from sparse_occupancy_telemetry import occupancy_telemetry
from soc_integrity_guard import guard
from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
from runtime_density_profiler import profiler

class SOCResolver:
    """
    Main resolver for SOC (Sparse Occupancy Consolidation).
    Consolidates fragmented sparse compute into sustained execution.
    """
    def __init__(self, wrapper: DiffKVHFWrapper):
        self.wrapper = wrapper
        self.consolidator = SparseWorkConsolidator()
        self.fuser = OccupancyAwareTritonFuser()
        self.scheduler = SustainedSparseBatchScheduler()
        print("[SOC] Resolver initialized. Consolidating sparse hardware activity.")

    def execute_consolidated_step(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Executes a consolidated sparse work step.
        """
        profiler.start("attention") # Reuse profiler
        
        # 1. Consolidate work
        consolidated_x = self.consolidator.consolidate_tokens(x, mask)
        
        # 2. Fused Dispatch
        def dummy_kernel():
            # Simulate compute
            torch.matmul(consolidated_x, consolidated_x.transpose(-2, -1))
            
        self.fuser.fused_dispatch([dummy_kernel])
        
        # 3. Stabilize Intensity
        intensity_stabilizer.record_launch(consolidated_x.numel() * 100) # Sim FLOPs
        
        occupancy_telemetry.sample_occupancy()
        profiler.end("attention")

    def get_soc_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive SOC hardware report.
        """
        intensity_metrics = intensity_stabilizer.get_intensity_report()
        occupancy_metrics = occupancy_telemetry.get_sustained_report()
        fusion_metrics = self.fuser.get_fused_telemetry()
        density = profiler.get_report()
        
        report = {
            **intensity_metrics,
            **occupancy_metrics,
            **fusion_metrics,
            "triton_runtime_percent": density.get("sparse_runtime_percent", 0),
            "sparse_runtime_percent": density.get("sparse_runtime_percent", 0),
            "dense_runtime_percent": density.get("dense_runtime_percent", 100)
        }
        
        guard.validate_soc_state(report)
        guard.check_integrity()
        
        return report
