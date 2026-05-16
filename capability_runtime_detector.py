
import torch
import os

class CapabilityRuntimeDetector:
    """
    Detects hardware and software capabilities of the current environment.
    """
    @staticmethod
    def detect():
        capabilities = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "nccl_available": torch.distributed.is_nccl_available() if hasattr(torch.distributed, 'is_nccl_available') else False,
            "peer_access": False,
            "cuda_graph_supported": True, # Assume true for modern CUDA
            "fast_fp16": False
        }
        
        if capabilities["gpu_count"] > 1:
            # Check for peer-to-peer access between GPU 0 and 1 as a proxy
            try:
                capabilities["peer_access"] = torch.cuda.can_device_access_peer(0, 1)
            except:
                capabilities["peer_access"] = False
                
        # Hardware specific checks
        if capabilities["cuda_available"]:
            prop = torch.cuda.get_device_properties(0)
            if prop.major >= 7: # Volta+ 
                capabilities["fast_fp16"] = True
                
        print(f"[CRMP] Capability Detection: {capabilities['gpu_count']} GPUs, P2P={capabilities['peer_access']}, NCCL={capabilities['nccl_available']}")
        return capabilities

if __name__ == "__main__":
    detector = CapabilityRuntimeDetector()
    print(detector.detect())
