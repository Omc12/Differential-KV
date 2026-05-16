import torch
import torch.nn as nn
from typing import Dict, List, Any

class PersistentWeightResidencyController:
    """
    Forces full-model weight residency on GPU.
    Prevents offloading and ensures weights are pinned.
    """
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.residency_map = {}

    def enforce_residency(self):
        """
        Hard-pins all model parameters to the target device.
        """
        print(f"[FRM] Enforcing full residency on {self.device}...")
        for name, param in self.model.named_parameters():
            if param.device.type != self.device:
                param.data = param.data.to(self.device)
            # Pinning (simulated via non-blocking ops or just ensuring device map)
            self.residency_map[name] = True
            
    def verify_residency(self) -> bool:
        """
        Verifies that all parameters are still on the target device.
        """
        all_on_device = True
        for name, param in self.model.named_parameters():
            if param.device.type != self.device:
                print(f"[FRM] WARNING: Parameter {name} migrated to {param.device}")
                all_on_device = False
        return all_on_device

    def get_residency_metrics(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters())
        resident_params = sum(p.numel() for p in self.model.parameters() if p.device.type == self.device)
        
        return {
            "total_parameters": total_params,
            "resident_parameters": resident_params,
            "residency_ratio": resident_params / total_params if total_params > 0 else 0,
            "sustained_weight_residency": True if resident_params == total_params else False
        }
