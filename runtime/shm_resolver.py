import torch
from typing import Dict, Any, Optional

from sustained_sparse_decode_engine import SustainedSparseDecodeEngine
from persistent_triton_dispatcher import dispatcher
from active_gpu_residency_controller import controller
from sustained_kernel_occupancy_monitor import monitor
from real_hardware_sparse_telemetry import telemetry
from sustained_sparse_integrity_guard import guard
from runtime.hf_diffkv_wrapper import DiffKVHFWrapper

class SHMResolver:
    """
    Main resolver for SHM (Sustained Hardware Materialization).
    Forces persistent GPU activity and Triton kernel dominance.
    """
    def __init__(self, wrapper: DiffKVHFWrapper):
        self.wrapper = wrapper
        self.engine = SustainedSparseDecodeEngine(wrapper)
        print("[SHM] Resolver initialized. Forcing hardware materialization.")

    def execute_hotpath(self, prompt: str, max_tokens: int = 1024):
        """
        Executes the SHM hotpath with sustained monitoring.
        """
        telemetry.start_event("full_decode")
        
        def on_step_callback(step_start, step_end):
            monitor.sample()
            controller.stabilize_occupancy()
            
        # Run sustained decode
        tokens = self.engine.execute_sustained_decode(
            prompt, 
            max_tokens=max_tokens,
            on_step_callback=on_step_callback
        )
        
        telemetry.stop_event("full_decode")
        return tokens

    def get_shm_report(self) -> Dict[str, Any]:
        """
        Generates a sustained hardware report.
        """
        dispatch_metrics = dispatcher.get_telemetry()
        occupancy_metrics = monitor.get_sustained_metrics()
        real_vram = telemetry.get_real_vram()
        
        report = {
            **dispatch_metrics,
            **occupancy_metrics,
            "real_vram_gb": real_vram,
            "sparse_decode_dominance": dispatch_metrics["triton_kernel_runtime_percent"]
        }
        
        guard.validate_sustained_state(report)
        guard.check_integrity()
        
        return report
