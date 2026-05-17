import torch
from typing import Dict, Any, List

class GpuResidencyRealityAuditor:
    """
    STAGE 4B.1.6 — ERCA GPU Residency Reality Auditor.
    Scans model parameters, dtypes, and devices to ensure total CUDA FP16 placement
    and verify physical VRAM residency bounds matching true 7B FP16 models.
    """
    def __init__(self):
        pass

    def audit_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Scans all parameters of the target PyTorch model and evaluates device residency.
        """
        total_params = 0
        cuda_params = 0
        cpu_params = 0
        fp16_params = 0
        other_dtype_params = 0

        param_devices = set()
        param_dtypes = set()

        for name, param in model.named_parameters():
            num_el = param.numel()
            total_params += num_el

            # Check device placement
            device_type = param.device.type
            param_devices.add(param.device)
            if device_type == "cuda":
                cuda_params += num_el
            else:
                cpu_params += num_el

            # Check dtype precision
            dtype = param.dtype
            param_dtypes.add(dtype)
            if dtype == torch.float16:
                fp16_params += num_el
            else:
                other_dtype_params += num_el

        total_elements = total_params
        cuda_ratio = cuda_params / total_elements if total_elements > 0 else 0.0
        fp16_ratio = fp16_params / total_elements if total_elements > 0 else 0.0

        # Parameter size in MB (FP16 = 2 bytes)
        param_memory_mb = (total_elements * 2) / (1024 * 1024)

        # Query physical CUDA allocated memory via torch APIs
        torch_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        torch_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

        passed = True
        violations = []

        if cuda_ratio < 0.999:
            passed = False
            violations.append(f"CUDA parameter ratio {cuda_ratio:.2%} < 99.9% (CPU fallback detected)")

        if fp16_ratio < 0.999:
            passed = False
            violations.append(f"FP16 parameter ratio {fp16_ratio:.2%} < 99.9% (Model precision mismatch)")

        # RTX 4070 SUPER has 12GB VRAM. Loading Qwen 7B FP16 requires ~14.5GB VRAM.
        # Active FP16 memory allocations will use Windows page faulting but must reside on GPU
        # In a fully resident state, torch.cuda.memory_allocated() must report at least 13.0 GB.
        if torch_allocated_mb < 13000.0:
            passed = False
            violations.append(f"Physical VRAM residency is too low: {torch_allocated_mb:.2f} MB (Expected >= 13000.0 MB for Qwen2.5-7B FP16). CPU offloading is strictly forbidden!")

        return {
            "passed": passed,
            "violations": violations,
            "total_parameters": total_elements,
            "cuda_parameters": cuda_params,
            "cpu_parameters": cpu_params,
            "fp16_parameters": fp16_params,
            "other_dtype_parameters": other_dtype_params,
            "cuda_ratio": cuda_ratio,
            "fp16_ratio": fp16_ratio,
            "param_memory_mb": param_memory_mb,
            "torch_allocated_vram_mb": torch_allocated_mb,
            "torch_reserved_vram_mb": torch_reserved_mb,
            "devices": [str(d) for d in param_devices],
            "dtypes": [str(dt) for dt in param_dtypes]
        }
