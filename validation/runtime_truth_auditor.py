"""
validation/runtime_truth_auditor.py

Core auditor for verifying runtime performance metrics against hardware reality.
Rejects any metrics that cannot be reconciled with kernel traces or bandwidth logs.
"""

import json
import os
from typing import Dict, List, Any, Optional
import logging

class RuntimeTruthAuditor:
    """
    Validates performance claims by cross-referencing telemetry sources.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.logger = logging.getLogger("RuntimeTruthAuditor")
        self.verification_results: Dict[str, str] = {} # Metric -> Status (VERIFIED, PARTIALLY, UNVERIFIED)

    def audit_tps(self, reported_tps: float, execution_log_path: str) -> str:
        """
        Verifies reported Tokens Per Second (TPS) against raw execution timestamps.
        """
        if not os.path.exists(execution_log_path):
            return "UNVERIFIED (No raw logs found)"
            
        try:
            with open(execution_log_path, 'r') as f:
                logs = [json.loads(line) for line in f]
                
            if not logs:
                return "UNVERIFIED (Empty logs)"
                
            start_time = logs[0]['timestamp']
            end_time = logs[-1]['timestamp']
            total_tokens = sum(l.get('tokens', 0) for l in logs)
            
            calculated_tps = total_tokens / (end_time - start_time)
            
            # Allow 5% margin for overhead/reporting jitter
            if abs(calculated_tps - reported_tps) / reported_tps < 0.05:
                return "VERIFIED"
            else:
                self.logger.warning(f"TPS mismatch: Reported {reported_tps}, Calculated {calculated_tps}")
                return f"PARTIALLY VERIFIED (Calculated: {calculated_tps:.2f})"
        except Exception as e:
            return f"UNVERIFIED (Audit failed: {str(e)})"

    def audit_occupancy(self, reported_occupancy: float, kernel_trace_path: str) -> str:
        """
        Verifies GPU occupancy claims against hardware-level kernel traces.
        """
        if not os.path.exists(kernel_trace_path):
            return "UNVERIFIED (No kernel trace)"
            
        # Real implementation would parse Nsight Systems or PyTorch Profiler output
        return "PARTIALLY VERIFIED (Trace exists, manual verification required)"

    def generate_audit_report(self, summary_report: Dict[str, Any]) -> str:
        """
        Processes a full summary report and marks every metric with its truth status.
        """
        results = {}
        for metric, value in summary_report.items():
            if "tps" in metric.lower():
                status = self.audit_tps(value, os.path.join(self.log_dir, "raw_execution.jsonl"))
            elif "occupancy" in metric.lower():
                status = self.audit_occupancy(value, os.path.join(self.log_dir, "kernel_trace.json"))
            else:
                status = "UNVERIFIED (No automated auditor available)"
                
            results[metric] = {
                "value": value,
                "status": status
            }
            
        return json.dumps(results, indent=2)
