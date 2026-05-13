"""
benchmarks/long_horizon_coding_tasks.py

Phase 12D: Long-Horizon Coding Tasks
Defines realistic coding tasks that require multi-hour sessions and 
deep repository knowledge.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CodingTask:
    id: str
    description: str
    files_involved: List[str]
    expected_output: str

LONG_HORIZON_TASKS = [
    CodingTask(
        id="refactor_distributed_kv",
        description="Refactor the distributed KV store to use the new hierarchical memory system across 15 files.",
        files_involved=["distributed/kv_store.py", "memory/hierarchical_sparse_memory.py"],
        expected_output="Functional hierarchical storage"
    ),
    CodingTask(
        id="debug_memory_leak",
        description="Find and fix a memory leak in the CUDA kernel orchestration loop by tracing state across 50k tokens of execution logs.",
        files_involved=["kernels/cuda_orch.cu", "runtime/executor.py"],
        expected_output="Zero memory growth over 1 hour"
    ),
    CodingTask(
        id="scale_to_million_tokens",
        description="Optimize the attention reinjection loop to handle 1.2M tokens of context without exceeding 24GB VRAM.",
        files_involved=["anchor_logic/semantic_anchor_system.py", "optimization/fast_reinjection.py"],
        expected_output="Successful 1.2M token generation"
    )
]

def get_tasks_by_complexity(min_files: int) -> List[CodingTask]:
    return [t for t in LONG_HORIZON_TASKS if len(t.files_involved) >= min_files]
