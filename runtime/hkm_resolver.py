"""
runtime/hkm_resolver.py

Unified Hardware Kernel Materialization (HKM) Resolver.
Orchestrates real hardware execution for Differential KV.
"""

import torch
import logging
from typing import Dict, Any, Optional

from hardware_materialization.triton_sparse_attention_materializer import TritonSparseAttentionMaterializer
from hardware_materialization.cuda_extension_sparse_ops import CUDASparseOps
from hardware_materialization.hardware_graph_capture_manager import HardwareGraphCaptureManager
from hardware_materialization.gpu_profiler_telemetry_bridge import GPUProfilerTelemetryBridge
from hardware_materialization.sparse_runtime_hotpath_extractor import SparseRuntimeHotpathExtractor
from hardware_materialization.hardware_materialization_integrity_guard import HardwareMaterializationIntegrityGuard

logger = logging.getLogger("HKMResolver")

class HKMResolver:
    """
    Main orchestration point for hardware-materialized runtime.
    Routes operations between real Triton/CUDA kernels and safe fallbacks.
    """
    def __init__(self, use_hardware: bool = True):
        self.use_hardware = use_hardware and torch.cuda.is_available()
        
        # Materialization Components
        self.triton_materializer = TritonSparseAttentionMaterializer()
        self.cuda_ops = CUDASparseOps()
        self.graph_manager = HardwareGraphCaptureManager()
        
        # Telemetry & Analysis
        self.telemetry = GPUProfilerTelemetryBridge()
        self.hotpath_extractor = SparseRuntimeHotpathExtractor()
        self.integrity_guard = HardwareMaterializationIntegrityGuard()

    def execute_sparse_attention(self, q, k, v, mask=None, validate: bool = False):
        """
        Executes sparse attention using the best available hardware path.
        """
        if not self.use_hardware:
            return self.triton_materializer._fallback(q, k, v, mask)

        self.telemetry.start_timer("triton_sparse_attn")
        
        # Hardware execution
        hw_out = self.triton_materializer.execute(q, k, v, mask)
        
        duration = self.telemetry.stop_timer("triton_sparse_attn")
        self.hotpath_extractor.trace_stage("triton_sparse_attn", duration)
        
        if validate:
            fb_out = self.triton_materializer._fallback(q, k, v, mask)
            self.integrity_guard.validate_outputs(hw_out, fb_out, "triton_sparse_attn")
            
        return hw_out

    def execute_reconstruction(self, u, v, anchor, indices=None, values=None, scale=1.0, validate: bool = False):
        """
        Executes low-rank + sparse reconstruction via CUDA-backed ops.
        """
        if not self.use_hardware:
            # Simple fallback
            out = torch.addmm(anchor, u, v, alpha=scale, beta=1.0)
            if indices is not None and indices.numel() > 0:
                out.view(-1).index_add_(0, indices.long().view(-1), values.view(-1).to(out.dtype))
            return out

        self.telemetry.start_timer("cuda_sparse_recon")
        
        hw_out = self.cuda_ops.fused_sparse_recon(u, v, anchor, indices, values, scale)
        
        duration = self.telemetry.stop_timer("cuda_sparse_recon")
        self.hotpath_extractor.trace_stage("cuda_sparse_recon", duration)
        
        if validate:
            # Re-run logic for fallback
            fb_out = torch.addmm(anchor, u, v, alpha=scale, beta=1.0)
            if indices is not None and indices.numel() > 0:
                fb_out.view(-1).index_add_(0, indices.long().view(-1), values.view(-1).to(fb_out.dtype))
            self.integrity_guard.validate_outputs(hw_out, fb_out, "cuda_sparse_recon")
            
        return hw_out

    def get_runtime_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive hardware telemetry."""
        metrics = self.telemetry.capture_telemetry()
        metrics["hotpaths"] = self.hotpath_extractor.get_bottlenecks()
        metrics["integrity"] = self.integrity_guard.get_summary()
        return metrics
