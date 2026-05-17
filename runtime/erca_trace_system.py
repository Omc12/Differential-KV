import os
import json
from pathlib import Path
from typing import Dict, Any

class ErcaTraceSystem:
    """
    STAGE 4B.1.6 — ERCA (Execution Reality Correlation Audit) Trace System.
    Dynamically writes the 10 mandated physical JSONL trace files to record 
    live execution parameters from the PyTorch model and NVML telemetry.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = {
            "full_transformer_execution": open(self.output_dir / "full_transformer_execution_trace.jsonl", "w", encoding="utf-8"),
            "layer_timing": open(self.output_dir / "layer_timing_trace.jsonl", "w", encoding="utf-8"),
            "cuda_kernel_launch": open(self.output_dir / "cuda_kernel_launch_trace.jsonl", "w", encoding="utf-8"),
            "operator_correlation": open(self.output_dir / "operator_correlation_trace.jsonl", "w", encoding="utf-8"),
            "vram_residency": open(self.output_dir / "vram_residency_trace.jsonl", "w", encoding="utf-8"),
            "parameter_placement": open(self.output_dir / "parameter_placement_trace.jsonl", "w", encoding="utf-8"),
            "power_draw": open(self.output_dir / "power_draw_trace.jsonl", "w", encoding="utf-8"),
            "nvml_telemetry": open(self.output_dir / "nvml_telemetry_trace.jsonl", "w", encoding="utf-8"),
            "logits_lineage": open(self.output_dir / "logits_lineage_trace.jsonl", "w", encoding="utf-8"),
            "token_reality": open(self.output_dir / "token_reality_trace.jsonl", "w", encoding="utf-8")
        }

    def write_record(self, trace_name: str, record: Dict[str, Any]):
        """
        Appends a record to the specified JSONL trace file.
        """
        if trace_name in self.files:
            f = self.files[trace_name]
            f.write(json.dumps(record) + "\n")
            f.flush()

    def close(self):
        """
        Safely closes all open trace file descriptors.
        """
        for name, f in self.files.items():
            try:
                f.close()
            except Exception:
                pass
