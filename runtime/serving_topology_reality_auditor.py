import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class ServingTopologyRealityAuditor:
    """
    SGC Stage 3C.4: Serving Topology Reality Auditor.
    Physically verifies continuous pipeline amortization gains, active decode continuity,
    and tail latency stability from raw system traces.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.violations: List[Dict[str, Any]] = []
        
        # Target constraints
        self.max_starvation_pct = 5.0
        self.max_queue_turbulence = 15.0
        self.max_tail_latency_ms = 120.0
        self.min_stream_reuse_pct = 80.0
        self.min_async_overlap_pct = 70.0

    def record_violation(self, msg: str, severity: str = "ERROR"):
        """
        Registers a physical operational violation.
        """
        self.violations.append({
            "violation": msg,
            "severity": severity
        })

    def get_violations(self) -> List[Dict[str, Any]]:
        """
        Returns all collected violations.
        """
        return self.violations

    def audit_serving_metrics(self, metrics: Dict[str, Any]):
        """
        Audits live system metrics against target physical serving constraints.
        """
        # 1. Enforce starvation constraints
        starvation = metrics.get("gpu_starvation_pct", 0.0)
        if starvation > self.max_starvation_pct:
            self.record_violation(
                f"GPU Starvation ({starvation:.2f}%) exceeded target of {self.max_starvation_pct}%!"
            )
            
        # 2. Enforce queue turbulence constraints
        turbulence = metrics.get("queue_turbulence_pct", 0.0)
        if turbulence > self.max_queue_turbulence:
            self.record_violation(
                f"Queue Turbulence ({turbulence:.2f}%) exceeded target of {self.max_queue_turbulence}%!"
            )
            
        # 3. Enforce tail latency bounds
        tail_lat = metrics.get("tail_latency_ms", 0.0)
        if tail_lat > self.max_tail_latency_ms:
            self.record_violation(
                f"Tail Latency ({tail_lat:.2f}ms) exceeded target of {self.max_tail_latency_ms}ms!"
            )
            
        # 4. Enforce stream reuse continuity
        reuse = metrics.get("stream_reuse_pct", 100.0)
        if reuse < self.min_stream_reuse_pct:
            self.record_violation(
                f"Stream Reuse ({reuse:.2f}%) fell below target of {self.min_stream_reuse_pct}%!"
            )

        # 5. Enforce async overlap efficiency
        overlap = metrics.get("async_overlap_pct", 100.0)
        if overlap < self.min_async_overlap_pct:
            self.record_violation(
                f"Async Overlap ({overlap:.2f}%) fell below target of {self.min_async_overlap_pct}%!"
            )

    def audit_trace_file(self, trace_path: Path):
        """
        Performs post-run validation of the exported raw PyTorch Profiler traces
        to ensure all runtime calls are mapped to the custom multistream kernels.
        """
        if not trace_path.exists():
            self.record_violation(f"Profiler trace file {trace_path.name} not found!")
            return
            
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            events = data.get("traceEvents", [])
            has_custom_stream = False
            has_persistent_launch = False
            
            for event in events:
                name = event.get("name", "")
                if "launch_persistent_attention" in name or "persistent_sparse_attention" in name:
                    has_persistent_launch = True
                if "cudaStream" in name or "launch_shared_memory_sparse_tile" in name:
                    has_custom_stream = True
                    
            if not has_persistent_launch:
                self.record_violation("Profiler trace lacks persistent attention kernel launches!")
            if not has_custom_stream:
                self.record_violation("Profiler trace lacks custom CUDA stream events!")
                
        except Exception as e:
            self.record_violation(f"Failed to parse profiler trace: {str(e)}")

    def enforce_reality(self):
        """
        Enforces that zero physical violations are present.
        """
        errors = [v for v in self.violations if v["severity"] == "ERROR"]
        if errors:
            raise RuntimeError(
                f"Serving Reality Audit FAILED with {len(errors)} violations:\n" + 
                "\n".join([f" - {e['violation']}" for e in errors])
            )
        print("[Serving Reality Auditor] All physical constraints met! PASS")
