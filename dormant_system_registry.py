
import os

class DormantSystemRegistry:
    """
    Tracks and manages distributed systems that remain packaged but inactive on single-GPU hardware.
    """
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.dormant_modules = [
            "distributed_kv_fabric",
            "cross_gpu_sync",
            "nccl_orchestration",
            "p2p_transfer_manager",
            "distributed_resonance_runtime"
        ]

    def register_dormancy(self):
        print(f"[CRMP] Registering {len(self.dormant_modules)} Distributed Systems as DORMANT")
        
        if self.capabilities["gpu_count"] <= 1:
            for module in self.dormant_modules:
                self._set_dormant(module)
        else:
            print("[CRMP] Multi-GPU detected. Distributed systems available but not auto-activated.")
            # Still keep them dormant by default unless explicitly needed
            for module in self.dormant_modules:
                self._set_dormant(module)

    def _set_dormant(self, module_name):
        env_var = f"DIFFKV_{module_name.upper()}_ACTIVE"
        os.environ[env_var] = "0"
        # We also set a sentinel for the runtime resolver
        os.environ[f"DIFFKV_{module_name.upper()}_DORMANT"] = "1"

    def get_status(self):
        return {m: os.environ.get(f"DIFFKV_{m.upper()}_ACTIVE", "0") == "1" for m in self.dormant_modules}
