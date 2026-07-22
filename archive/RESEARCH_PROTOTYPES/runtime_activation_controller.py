
from capability_runtime_detector import CapabilityRuntimeDetector
from optimization_hotpath_materializer import OptimizationHotpathMaterializer
from dormant_system_registry import DormantSystemRegistry
import os

class RuntimeActivationController:
    """
    Orchestrates the activation of optimizations and the dormancy of distributed systems.
    """
    def __init__(self):
        self.detector = CapabilityRuntimeDetector()
        self.capabilities = self.detector.detect()
        
        self.materializer = OptimizationHotpathMaterializer(self.capabilities)
        self.registry = DormantSystemRegistry(self.capabilities)

    def activate(self):
        print("=========================================================")
        print("CRMP — Critical Runtime Materialization Pass")
        print("=========================================================")
        
        # 1. Materialize local hotpath
        active_opts = self.materializer.materialize()
        
        # 2. Register distributed dormancy
        self.registry.register_dormancy()
        
        # 3. Final system state
        os.environ["DKV_CRMP_ACTIVE"] = "1"
        
        print(f"\\n[CRMP] System stabilized with {len(active_opts)} active optimizations.")
        print("=========================================================\\n")
        
        return {
            "capabilities": self.capabilities,
            "active_optimizations": active_opts,
            "dormant_systems": self.registry.get_status()
        }

if __name__ == "__main__":
    controller = RuntimeActivationController()
    controller.activate()
