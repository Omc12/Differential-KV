
import os
import torch
from typing import Dict, Any

class CRMPResolver:
    """
    Runtime resolver for capability-gated activation and materialization.
    """
    def __init__(self):
        # We check if CRMP was already activated by the controller
        self.is_materialized = os.environ.get("DIFFKV_CRMP_ACTIVE") == "1"
        
    def resolve_path(self, op_type: str) -> bool:
        """
        Determines if a specific optimization or system should be active.
        """
        if op_type == "distributed":
            # Hard-gated: always False on single GPU or if CRMP says so
            return os.environ.get("DIFFKV_DISTRIBUTED_KV_FABRIC_ACTIVE") == "1"
            
        if not self.is_materialized:
            return False
            
        # Map op_type to environment variables
        mapping = {
            "cuda_graph": "DIFFKV_USE_CUDA_GRAPHS",
            "fusion": "DIFFKV_FUSE_KERNELS",
            "hbm": "DIFFKV_HBM_OPTIMIZATION",
            "deterministic": "DIFFKV_DETERMINISTIC_MICROBATCH",
            "sparse_schedule": "DIFFKV_SPARSE_SCHEDULING"
        }
        
        env_var = mapping.get(op_type)
        if env_var:
            return os.environ.get(env_var) == "1"
            
        return False

    def get_runtime_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary of active optimizations for kernel injection.
        """
        return {
            "use_cuda_graph": self.resolve_path("cuda_graph"),
            "use_fusion": self.resolve_path("fusion"),
            "use_hbm_opt": self.resolve_path("hbm"),
            "deterministic": self.resolve_path("deterministic"),
            "sparse_schedule": self.resolve_path("sparse_schedule"),
            "mode": "single-gpu-materialized" if self.is_materialized else "baseline"
        }
