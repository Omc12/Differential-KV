import os
import torch
from pathlib import Path

class RawTorchProfilerRecorder:
    """
    RHD Phase 41.4.6 — Raw Torch Profiler Recorder.
    Captures PyTorch operator execution, CUDA kernels, GEMMs, memory allocations,
    and CUDA synchronizations using torch.profiler.
    Saves a raw chrome-compatible JSON trace.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "telemetry/stage3b/phase_41_4_6_rhd"
        self.trace_path = self.trace_dir / "raw_torch_profiler_trace.json"
        self.profiler = None

    def start(self):
        os.makedirs(self.trace_dir, exist_ok=True)
        # Initialize the native PyTorch profiler
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True
        )
        self.profiler.__enter__()
        print("[Torch Profiler] Profiling started.")

    def step(self):
        if self.profiler:
            self.profiler.step()

    def stop(self):
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            # Export raw chrome trace format
            self.profiler.export_chrome_trace(str(self.trace_path))
            print(f"[Torch Profiler] Trace exported raw to: {self.trace_path}")
            self.profiler = None
