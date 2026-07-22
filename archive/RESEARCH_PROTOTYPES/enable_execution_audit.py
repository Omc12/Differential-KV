import sys
import os
import json
import torch
import time
from typing import Dict, Any, List

class ExecutionAuditor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExecutionAuditor, cls).__new__(cls)
            cls._instance.config = {
                "trace_attention": False,
                "trace_kernels": False,
                "trace_kv": False,
                "trace_virtualization": False,
                "trace_triton": False,
                "trace_cuda_graphs": False,
                "trace_memory": False,
                "trace_fallbacks": False,
            }
            cls._instance.trace_log = []
            cls._instance.export_path = "telemetry/execution_trace.json"
        return cls._instance

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            elif key == "export_trace":
                self.export_path = value

    def log_event(self, category: str, event_type: str, data: Dict[str, Any]):
        # Always log if category matches config or if it's a fallback and trace_fallbacks is on
        enabled = self.config.get(f"trace_{category}", False) or (category == "fallbacks" and self.config.get("trace_fallbacks", False))
        
        if not enabled:
            return

        event = {
            "timestamp": time.time(),
            "category": category,
            "event_type": event_type,
            "data": data,
            "vram_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        self.trace_log.append(event)
        
        # Also print to stdout for real-time visibility as requested
        print(f"[AUDIT][{category.upper()}] {event_type}: {data}")

    def export(self):
        if not self.trace_log:
            return
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
        with open(self.export_path, 'w') as f:
            json.dump(self.trace_log, f, indent=2)
        print(f"Execution trace exported to {self.export_path}")

# Global instance
auditor = ExecutionAuditor()

def patch_runtime():
    """
    Monkey-patch runtime components to insert audit hooks.
    """
    # 1. Patch TritonDKV
    try:
        from runtime.triton_dkv import TritonDKV
        
        original_recon = TritonDKV.reconstruct_lowrank
        def audited_recon(U, V, anchor, scale=1.0):
            try:
                from runtime.triton_dkv import triton_fused_reconstruct
                # Force Triton check
                auditor.log_event("triton", "launch", {"kernel": "lowrank_recon_kernel", "shape": list(U.shape)})
                res = triton_fused_reconstruct(U, V, anchor, scale=scale)
                auditor.log_event("kernels", "custom_kernel_success", {"kernel": "triton_lowrank"})
                return res
            except Exception as e:
                auditor.log_event("fallbacks", "triton_failure", {"error": str(e), "path": "lowrank_recon"})
                # Fallback implementation
                return (torch.matmul(U.float(), V.float()) * scale + anchor.float()).to(U.dtype)
        
        TritonDKV.reconstruct_lowrank = audited_recon
        TritonDKV.reconstruct_lowrank_sparse = TritonDKV.reconstruct_lowrank_sparse # Ensure it uses the patched version if it calls it
    except ImportError:
        print("Warning: Could not patch TritonDKV")

    # 2. Patch NativeSparseAttention
    try:
        from runtime.native_sparse_attention import NativeSparseAttention
        original_sparse_forward = NativeSparseAttention.forward
        def audited_sparse_forward(self, q, k, v, curvature, mask=None):
            auditor.log_event("attention", "sparse_forward_start", {"seq_len": k.shape[2], "sparse_ratio": self.sparse_ratio})
            res = original_sparse_forward(self, q, k, v, curvature, mask)
            auditor.log_event("attention", "sparse_forward_end", {"output_shape": list(res.shape)})
            return res
        NativeSparseAttention.forward = audited_sparse_forward
    except ImportError:
        print("Warning: Could not patch NativeSparseAttention")

    # 3. Patch CUDAGraphExecution
    try:
        from runtime.cuda_graph_execution import CUDAGraphExecution
        original_replay = CUDAGraphExecution.replay
        def audited_replay(self):
            auditor.log_event("cuda_graphs", "replay_start", {})
            res = original_replay(self)
            if res is not None:
                auditor.log_event("cuda_graphs", "replay_success", {})
            else:
                auditor.log_event("cuda_graphs", "replay_inactive", {})
            return res
        CUDAGraphExecution.replay = audited_replay
    except ImportError:
        print("Warning: Could not patch CUDAGraphExecution")

    # 4. Patch KVRuntimeManager
    try:
        from runtime.kv_runtime_manager import KVRuntimeManager
        original_add_block = KVRuntimeManager.add_block
        def audited_add_block(self, layer_idx, block):
            auditor.log_event("kv", "add_block", {"layer": layer_idx, "mode": block.mode, "is_compressed": block.is_compressed})
            return original_add_block(self, layer_idx, block)
        KVRuntimeManager.add_block = audited_add_block
        
        original_reconstruct = KVRuntimeManager.reconstruct_layer
        def audited_reconstruct(self, layer_idx, target_dtype=torch.float16):
            auditor.log_event("kv", "reconstruct_layer_start", {"layer": layer_idx})
            res = original_reconstruct(self, layer_idx, target_dtype)
            auditor.log_event("kv", "reconstruct_layer_end", {"layer": layer_idx})
            return res
        KVRuntimeManager.reconstruct_layer = audited_reconstruct
    except ImportError:
        print("Warning: Could not patch KVRuntimeManager")

    print("Runtime components patched for auditing.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-attention", action="store_true")
    parser.add_argument("--trace-kernels", action="store_true")
    parser.add_argument("--trace-kv", action="store_true")
    parser.add_argument("--trace-virtualization", action="store_true")
    parser.add_argument("--trace-triton", action="store_true")
    parser.add_argument("--trace-cuda-graphs", action="store_true")
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--trace-fallbacks", action="store_true")
    parser.add_argument("--export-trace", type=str, default="telemetry/execution_trace.json")
    
    args = parser.parse_args()
    
    # This part is just for verification when run directly
    auditor.configure(
        trace_attention=args.trace_attention,
        trace_kernels=args.trace_kernels,
        trace_kv=args.trace_kv,
        trace_virtualization=args.trace_virtualization,
        trace_triton=args.trace_triton,
        trace_cuda_graphs=args.trace_cuda_graphs,
        trace_memory=args.trace_memory,
        trace_fallbacks=args.trace_fallbacks,
        export_trace=args.export_trace
    )
    
    print(f"Auditor configured. Settings: {auditor.config}")
