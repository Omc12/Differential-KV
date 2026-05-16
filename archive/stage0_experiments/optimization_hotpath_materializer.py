
import os
import torch

class OptimizationHotpathMaterializer:
    """
    Materializes local single-GPU optimizations into the active inference path.
    """
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.active_optimizations = []

    def materialize(self):
        print("[CRMP] Materializing Local Optimization Hotpath...")
        
        # 1. CUDA Graph Replay
        if self.capabilities["cuda_graph_supported"]:
            self._activate_cuda_graphs()
            
        # 2. Kernel Fusion
        self._activate_kernel_fusion()
        
        # 3. Occupancy & HBM Optimization
        self._optimize_memory_traffic()
        
        # 4. Deterministic Replay
        self._ensure_deterministic_replay()
        
        # 5. Sparse Scheduling
        self._activate_sparse_scheduling()

        return self.active_optimizations

    def _activate_cuda_graphs(self):
        os.environ["DIFFKV_USE_CUDA_GRAPHS"] = "1"
        os.environ["DIFFKV_GRAPH_PERSISTENCE"] = "1"
        self.active_optimizations.append("cuda_graph_replay")
        print("  -> CUDA Graph Replay: ENABLED")

    def _activate_kernel_fusion(self):
        os.environ["DIFFKV_FUSE_KERNELS"] = "1"
        os.environ["DIFFKV_TRITON_AUTO_TUNE"] = "1"
        self.active_optimizations.append("runtime_kernel_fusion")
        print("  -> Runtime Kernel Fusion: ENABLED")

    def _optimize_memory_traffic(self):
        os.environ["DIFFKV_HBM_OPTIMIZATION"] = "1"
        os.environ["DIFFKV_OCCUPANCY_STABILIZATION"] = "1"
        self.active_optimizations.append("hbm_traffic_optimization")
        print("  -> HBM Traffic Optimization: ENABLED")

    def _ensure_deterministic_replay(self):
        os.environ["DIFFKV_DETERMINISTIC_MICROBATCH"] = "1"
        torch.use_deterministic_algorithms(False) # We use custom determinism
        self.active_optimizations.append("deterministic_microbatching")
        print("  -> Deterministic Microbatching: ENABLED")

    def _activate_sparse_scheduling(self):
        os.environ["DIFFKV_SPARSE_SCHEDULING"] = "1"
        self.active_optimizations.append("sparse_scheduling")
        print("  -> Sparse Scheduling: ENABLED")
