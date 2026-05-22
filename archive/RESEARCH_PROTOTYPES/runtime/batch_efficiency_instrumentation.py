import time
import json
from typing import Dict, List, Any

class BatchEfficiencyInstrumentation:
    """
    STAGE 2 DQO: Batch Efficiency Instrumentation.
    Tracks REAL batch size, overlap ratio, and queue-to-batch conversion efficiency.
    NO inferred percentages. Raw measurements only.
    """
    def __init__(self, trace_path: str = "traces/stage2/phase_38_8_dqo/live_batch_efficiency.jsonl"):
        self.trace_path = trace_path
        self.batch_sizes: List[int] = []
        self.queue_depths_at_batch: List[int] = []
        self.active_overlaps: List[int] = []
        self.step_durations: List[float] = []
        
    def record_batch_step(self, batch_size: int, queue_depth: int, active_overlap: int, duration_sec: float):
        self.batch_sizes.append(batch_size)
        self.queue_depths_at_batch.append(queue_depth)
        self.active_overlaps.append(active_overlap)
        self.step_durations.append(duration_sec)
        
        self._log_efficiency(batch_size, queue_depth, active_overlap, duration_sec)

    def _log_efficiency(self, size: int, depth: int, overlap: int, duration: float):
        entry = {
            "timestamp": time.time(),
            "batch_size": size,
            "queue_depth": depth,
            "active_overlap": overlap,
            "step_duration_ms": duration * 1000
        }
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_efficiency_metrics(self) -> Dict[str, Any]:
        avg_batch = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        avg_queue = sum(self.queue_depths_at_batch) / len(self.queue_depths_at_batch) if self.queue_depths_at_batch else 0
        
        # conversion efficiency: batch_size / (batch_size + queue_depth)
        # Higher means we are clearing the queue effectively per step
        total_potential = [s + d for s, d in zip(self.batch_sizes, self.queue_depths_at_batch)]
        avg_conv_eff = sum(s/p if p > 0 else 1.0 for s, p in zip(self.batch_sizes, total_potential)) / len(self.batch_sizes) if self.batch_sizes else 1.0

        return {
            "avg_real_batch_size": avg_batch,
            "avg_queue_depth": avg_queue,
            "queue_to_batch_efficiency": avg_conv_eff,
            "total_steps_instrumented": len(self.batch_sizes)
        }
