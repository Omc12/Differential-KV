"""
profiling/runtime_memory_profiler.py

Tracks actual VRAM usage across different runtimes and context lengths.
Supports CPU fallback monitoring.
"""

import torch
import psutil
import os
from typing import Dict, Any

class RuntimeMemoryProfiler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def get_vram_usage(self) -> float:
        """Returns VRAM usage in MB."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
        
    def get_system_ram_usage(self) -> float:
        """Returns system RAM usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def profile_session(self, adapter: Any, context_len: int) -> Dict[str, float]:
        """
        Profiles a full inference session.
        """
        initial_vram = self.get_vram_usage()
        initial_ram = self.get_system_ram_usage()
        
        # Trigger inference
        adapter.generate("Memory test. " * (context_len // 10), max_tokens=1)
        
        final_vram = self.get_vram_usage()
        final_ram = self.get_system_ram_usage()
        
        return {
            "vram_delta_mb": final_vram - initial_vram,
            "system_ram_delta_mb": final_ram - initial_ram,
            "vram_total_mb": final_vram
        }
