import os
from pathlib import Path
from runtime.raw_nvidia_smi_capture_system import RawNvidiaSmiCaptureSystem
from runtime.raw_torch_profiler_recorder import RawTorchProfilerRecorder
from runtime.raw_cuda_event_trace_layer import RawCudaEventTraceLayer
from runtime.real_vram_allocation_recorder import RealVramAllocationRecorder
from runtime.real_transformer_activity_recorder import RealTransformerActivityRecorder
from runtime.gpu_timeline_dump_system import GpuTimelineDumpSystem

class RawHardwareTraceSystem:
    """
    RHD Phase 41.4.6 — Raw Hardware Trace System.
    Orchestrates the lifecycle of all raw hardware and transformer trace recorders.
    Enforces raw data logging only. Absolutely no summaries, metrics, or conclusions.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        
        # Initialize recorders
        self.nvidia_smi = RawNvidiaSmiCaptureSystem(workspace_root)
        self.profiler = RawTorchProfilerRecorder(workspace_root)
        self.cuda_events = RawCudaEventTraceLayer(workspace_root)
        self.vram = RealVramAllocationRecorder(workspace_root)
        self.transformer = RealTransformerActivityRecorder(workspace_root)
        self.timeline = GpuTimelineDumpSystem(workspace_root)

    def start_recording(self):
        """Starts all system and torch-level telemetry tools."""
        print("[Raw Hardware Trace] Starting all hardware trace recorders...")
        self.nvidia_smi.start()
        self.profiler.start()
        print("[Raw Hardware Trace] Recorders are fully active.")

    def step(self):
        """Advances profiler steps if needed."""
        self.profiler.step()

    def stop_recording(self):
        """Stops all system recorders and flushes raw telemetry to disk."""
        print("[Raw Hardware Trace] Stopping all hardware trace recorders...")
        self.nvidia_smi.stop()
        self.profiler.stop()
        self.cuda_events.synchronize_and_flush()
        print("[Raw Hardware Trace] Telemetry traces successfully flushed to disk.")
