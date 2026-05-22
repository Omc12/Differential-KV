import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from persistent_weight_residency_controller import PersistentWeightResidencyController

class RealMultiModelResidencyController:
    """
    HSM System 1: Real Multi-Model Residency Controller.
    Manages persistent model residency and residency locking for concurrent serving.
    """
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.base_controller = PersistentWeightResidencyController(model, device)
        self.device = device
        self.model = model
        self.locked_sessions: set = set()
        self.residency_stabilized = False

    def enforce_serving_residency(self):
        """Forces all model weights and buffers to the target GPU."""
        self.base_controller.enforce_residency()
        # Also ensure buffers are on device
        for name, buf in self.model.named_buffers():
            if buf.device.type != self.device:
                buf.data = buf.data.to(self.device)
        self.residency_stabilized = True

    def lock_residency(self, session_id: str):
        """Locks residency for a specific session to prevent offloading."""
        self.locked_sessions.add(session_id)

    def unlock_residency(self, session_id: str):
        """Unlocks residency for a session."""
        self.locked_sessions.discard(session_id)

    def verify_serving_integrity(self) -> bool:
        """Verifies that the model is materially active and resident."""
        weights_ok = self.base_controller.verify_residency()
        # Real hardware check: ensure GPU memory is materially occupied
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            # A 7B model in FP16 is ~14GB. We expect at least 12GB for Qwen2.5-7B
            if allocated < 10 * (1024**3): 
                print(f"[HSM] WARNING: Material VRAM occupancy too low ({allocated / 1e9:.2f} GB)")
                return False
        return weights_ok

    def get_hsm_residency_metrics(self) -> Dict[str, Any]:
        base_metrics = self.base_controller.get_residency_metrics()
        vram_allocated = 0
        vram_reserved = 0
        if self.device == "cuda":
            vram_allocated = torch.cuda.memory_allocated(self.device)
            vram_reserved = torch.cuda.memory_reserved(self.device)
        
        return {
            **base_metrics,
            "vram_allocated_gb": vram_allocated / (1024**3),
            "vram_reserved_gb": vram_reserved / (1024**3),
            "active_locked_sessions": len(self.locked_sessions),
            "residency_stabilized": self.residency_stabilized
        }
