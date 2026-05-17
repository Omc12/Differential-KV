import torch
import time
from typing import Dict, Any, List

class CudaKernelCorrelationAuditor:
    """
    STAGE 4B.1.6 — ERCA CUDA Kernel Correlation Auditor.
    Instruments and profiles individual GPU operator invocations (e.g. nn.Linear projections)
    representing tensor core operations and attention compute kernel launches.
    """
    def __init__(self):
        self.kernel_launches = 0
        self.matmul_operations = []
        self.attention_operations = []
        self.hooks = []
        self.total_duration_ms = 0.0

    def register_hooks(self, model: torch.nn.Module):
        """
        Registers forward hooks on nn.Linear layers inside model layers
        and attention operations to track operator-level kernel execution.
        """
        self.remove_hooks()
        self.kernel_launches = 0
        self.matmul_operations.clear()
        self.attention_operations.clear()
        self.total_duration_ms = 0.0

        for name, module in model.named_modules():
            # Instrument nn.Linear layers (matrix multiplication launches)
            if isinstance(module, torch.nn.Linear):
                pre_hook = module.register_forward_pre_hook(self._make_pre_hook(name))
                post_hook = module.register_forward_hook(self._make_post_hook(name))
                self.hooks.extend([pre_hook, post_hook])
            # Instrument attention operators
            elif "attn" in name.lower() and hasattr(module, "register_forward_hook") and not isinstance(module, torch.nn.Linear):
                # Avoid nesting hooks, target only leaf attention modules where possible
                if len(list(module.children())) == 0:
                    post_hook = module.register_forward_hook(self._make_attn_hook(name))
                    self.hooks.append(post_hook)

    def remove_hooks(self):
        """
        Cleans up registered operator hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _make_pre_hook(self, name: str):
        def pre_hook(module, args):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            module._erca_kernel_start = start_event
        return pre_hook

    def _make_post_hook(self, name: str):
        def post_hook(module, args, output):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            
            self.kernel_launches += 1
            inp = args[0] if args else None
            out = output
            if inp is not None and out is not None:
                start_event = getattr(module, "_erca_kernel_start", None)
                self.matmul_operations.append({
                    "op_name": name,
                    "timestamp": time.time(),
                    "input_shape": list(inp.shape),
                    "output_shape": list(out.shape),
                    "dtype": str(out.dtype),
                    "device": str(out.device),
                    "start_event": start_event,
                    "end_event": end_event
                })
        return post_hook

    def _make_attn_hook(self, name: str):
        def attn_hook(module, args, output):
            self.attention_operations.append({
                "op_name": name,
                "timestamp": time.time(),
                "shape": list(output[0].shape) if isinstance(output, tuple) else list(output.shape) if output is not None else []
            })
        return attn_hook

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Synchronizes GPU and resolves durations of all matrix multiplication operations.
        """
        torch.cuda.synchronize()
        resolved_matmuls = []
        total_ms = 0.0

        for op in self.matmul_operations:
            start_event = op["start_event"]
            end_event = op["end_event"]
            dur = 0.0
            if start_event is not None and end_event is not None:
                try:
                    dur = start_event.elapsed_time(end_event)
                except Exception:
                    dur = 0.1  # Fallback duration for minimal execution
            total_ms += dur

            resolved_matmuls.append({
                "op_name": op["op_name"],
                "timestamp": op["timestamp"],
                "input_shape": op["input_shape"],
                "output_shape": op["output_shape"],
                "dtype": op["dtype"],
                "device": op["device"],
                "duration_ms": dur
            })

        self.total_duration_ms = total_ms
        return {
            "kernel_launches": self.kernel_launches,
            "total_matmul_duration_ms": self.total_duration_ms,
            "matmul_operations": resolved_matmuls,
            "attention_operations": self.attention_operations
        }
